"""
Train a meta-VSI-SR model by multiple GPU
Created on Tue Sep 30 19:31:36 2023
@ Last Updated by Lin Yuhuan
"""
import os
import time
from Code_for_2D_IsoRecon.utils.utils import dataset_loader
import numpy as np
import torch
import tifffile as tiff
from Code_for_2D_IsoRecon.utils.utils import XxPrctileNorm, save_model, save_checkpoint, print_time
from torch.utils.tensorboard import SummaryWriter
from model import meta_SR_2D_GAN
from Code_for_2D_IsoRecon.utils.utils import utils
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist

# about multiprocessing GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3, 6'
nprocs = 3

parser = argparse.ArgumentParser()
# dataset for training, validating and testing
parser.add_argument("--Data_path", type=str, default='/Code_for_2D_IsoRecon/data/train')
parser.add_argument("--filter_flag", type=bool, default=True)
parser.add_argument("--filter_sigma", type=float, default=0.6)
# parameters of meta learning
parser.add_argument("--meta_batch_size", type=int, default=3)
parser.add_argument("--meta_iter", type=int, default=100000)
parser.add_argument("--task_batch_size", type=int, default=8)
parser.add_argument("--spt_num", type=int, default=4)
parser.add_argument("--task_iter", type=int, default=3)
parser.add_argument("--g_lr_out", type=float, default=1e-4)
parser.add_argument("--d_lr_out", type=float, default=2e-5)
parser.add_argument("--g_lr_in", type=float, default=1e-2)
parser.add_argument("--d_lr_in", type=float, default=2e-3)
parser.add_argument("--spt_num_val", type=int, default=5)
parser.add_argument("--val_num", type=int, default=20)
parser.add_argument("--second_order", type=bool, default=False)
# save path
parser.add_argument("--save_step", type=int, default=1000)
parser.add_argument("--print_step", type=int, default=20)
parser.add_argument("--load_checkpoint_flag", type=bool, default=True)
parser.add_argument("--Save_path", type=str, default='/Code_for_2D_IsoRecon/meta_rcan')
# about model
parser.add_argument("--in_channel", type=int, default=7)
parser.add_argument("--out_channel", type=int, default=3)
parser.add_argument("--num_resgroup", type=int, default=4)
parser.add_argument("--num_resblock", type=int, default=4)
# about loss fn
parser.add_argument("--fft_param", type=float, default=1.0)
parser.add_argument("--ssim_param", type=float, default=1e-1)
args = parser.parse_args()

Source_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main(nprocs, args):
    mp.spawn(main_worker, nprocs=nprocs, args=(nprocs, args))


def main_worker(local_rank, nprocs, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    dist.init_process_group(backend='nccl', world_size=nprocs, rank=local_rank)
    Trainer = Train(args=args, nprocs=nprocs, local_rank=local_rank)
    Trainer()


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


class Train(object):

    def __init__(self, args, nprocs, local_rank):
        """Train a meta-VSI-SR model by multiple GPU"""
        print('[*] Initialize Training')

        '''hyperparameters'''
        self.data_path = Source_path+args.Data_path + "/meta_train"

        self.step = 0

        self.meta_iter = args.meta_iter

        self.task_batch_size = args.task_batch_size
        self.task_iter = args.task_iter

        self.spt = args.spt_num
        self.spt_val = args.spt_num_val
        self.val_path = Source_path+args.Data_path + "/meta_val"
        self.val_num = args.val_num

        self.g_lr_out = args.g_lr_out
        self.d_lr_out = args.d_lr_out
        self.g_lr_in = args.g_lr_in
        self.d_lr_in = args.d_lr_in

        self.train_g_times = 3
        self.train_d_times = 1

        self.second_order = args.second_order

        self.test_path = Source_path+args.Data_path + "/meta_test/input"
        self.test_save_path = Source_path+args.Save_path + '/test'

        self.checkpoint_dir = Source_path+args.Save_path + '/model'
        self.log_dir = Source_path+args.Save_path + '/log'

        self.print_iter = args.print_step
        self.save_iter = args.save_step
        self.filter_flag = args.filter_flag
        self.filter_sigma = args.filter_sigma

        self.meta_batch_size = args.meta_batch_size // nprocs
        self.nprocs = nprocs
        self.local_rank = local_rank

        '''GPU'''
        torch.cuda.set_device(local_rank)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # torch.cuda.set_per_process_memory_fraction(0.7, 0)

        '''model'''
        self.model = meta_SR_2D_GAN.Generator(in_channel=args.in_channel, out_channel=args.out_channel,
                                              n_ResGroup=args.num_resgroup, n_RCAB=args.num_resblock)
        self.discriminator = meta_SR_2D_GAN.Discriminator(in_ch=args.out_channel)

        '''loading checkpoint'''
        if args.load_checkpoint_flag:
            if os.path.exists(self.checkpoint_dir + '/' + 'g_latest.pth'):
                checkpoint = torch.load(self.checkpoint_dir + '/' + 'g_latest.pth')
                self.step = checkpoint['step']
                self.model.load_state_dict(checkpoint['model'])
                # self.g_opt.load_state_dict(checkpoint['optim'])
                print('Loading gmodel %d' % self.step)

            if os.path.exists(self.checkpoint_dir + '/' + 'd_latest.pth'):
                checkpoint = torch.load(self.checkpoint_dir + '/' + 'd_latest.pth')
                self.discriminator.load_state_dict(checkpoint['model'])
                # self.d_opt.load_state_dict(checkpoint['optim'])
                print('Loading dmodel %d' % self.step)

        self.model = self.model.cuda(local_rank)
        self.discriminator = self.discriminator.cuda(local_rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
        self.discriminator = torch.nn.parallel.DistributedDataParallel(self.discriminator, device_ids=[local_rank])

        '''loss'''
        self.d_loss_fn = torch.nn.BCELoss().cuda(local_rank)
        self.g_loss_fn = utils.loss_mse_ssim_fft(device='cuda:{}'.format(local_rank), ssim_param=args.ssim_param, fft_param=args.fft_param)

        '''Optimizers'''
        self.g_opt = torch.optim.Adam(self.model.parameters(), lr=self.g_lr_out)
        self.d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_lr_out)

        '''lr scheduler'''
        self.g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.g_opt, mode='min', factor=0.5, patience=10,
                                                                      threshold=1e-4,
                                                                      cooldown=0, verbose=True,
                                                                      min_lr=self.g_lr_out * 0.01)
        self.d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.d_opt, mode='min', factor=0.5, patience=10,
                                                                      threshold=1e-4,
                                                                      cooldown=0, verbose=True,
                                                                      min_lr=self.d_lr_out * 0.01)
        # self.g_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.g_opt, [10000, 40000, 60000], gamma=0.5)
        # self.d_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.d_opt, [10000, 40000, 60000], gamma=0.5)

        '''Tensorboard'''
        if local_rank == 0:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)

        self.d_total_lossesb, self.g_total_lossesb, self.inner_loop_pre, self.inner_loop_aft = None, None, None, None
        self.g_weighted_total_lossesb, self.d_weighted_total_lossesb = None, None
        self.reduced_lossg, self.reduced_lossd = None, None

    def construct_model(self, inf):
        def task_meta_learning(inputa, labela, inputb, labelb, valid, fake, valid1, fake1):

            d_task_outputsb, d_task_lossesb = [], []
            g_task_outputsb, g_task_lossesb = [], []

            # -------------------
            # Train discriminator
            # -------------------
            # loss before discriminator updated
            d_real_loss = self.d_loss_fn(self.discriminator(labelb, vars=None), valid1)
            self.gen_imgs = self.model(inputb, vars=None)
            d_fake_loss = self.d_loss_fn(self.discriminator(self.gen_imgs.detach(), vars=None), fake1)
            self.d_loss = (d_real_loss + d_fake_loss) / 2
            d_task_lossesb.append(self.d_loss)

            d_real_loss = self.d_loss_fn(self.discriminator(labela, vars=None), valid)
            self.gen_imgs = self.model(inputa, vars=None)
            d_fake_loss = self.d_loss_fn(self.discriminator(self.gen_imgs.detach(), vars=None), fake)
            self.d_loss = (d_real_loss + d_fake_loss) / 2
            d_grad = torch.autograd.grad(self.d_loss, self.discriminator.module.parameters(),
                                         create_graph=self.second_order)
            d_weights = list(
                map(lambda p: p[1] - self.d_lr_in * p[0], zip(d_grad, self.discriminator.module.parameters())))
            for j in range(self.train_d_times - 1):
                d_real_loss = self.d_loss_fn(self.discriminator(labela, vars=d_weights), valid)
                self.gen_imgs = self.model(inputa, vars=None)
                d_fake_loss = self.d_loss_fn(self.discriminator(self.gen_imgs.detach(), vars=d_weights), fake)
                self.d_loss = (d_real_loss + d_fake_loss) / 2
                d_grad = torch.autograd.grad(self.d_loss, d_weights, create_graph=self.second_order)
                d_weights = list(map(lambda p: p[1] - self.d_lr_in * p[0], zip(d_grad, d_weights)))
            # 1st inner loss for discriminator
            d_real_loss = self.d_loss_fn(self.discriminator(labelb, vars=d_weights), valid1)
            self.gen_imgs = self.model(inputb, vars=None)
            d_fake_loss = self.d_loss_fn(self.discriminator(self.gen_imgs.detach(), vars=d_weights), fake1)
            self.d_loss = (d_real_loss + d_fake_loss) / 2
            d_task_lossesb.append(self.d_loss)

            # ---------------
            # Train generator
            # ---------------
            # loss before generator updated
            self.gen_imgs = self.model(inputb, vars=None)
            g_loss1 = self.g_loss_fn(labelb, self.gen_imgs)
            g_loss2 = self.d_loss_fn(self.discriminator(self.gen_imgs, vars=d_weights), valid1)
            self.g_loss = g_loss1 + 0.1 * g_loss2
            g_task_lossesb.append(self.g_loss)

            self.gen_imgs = self.model(inputa, vars=None)
            g_loss1 = self.g_loss_fn(labela, self.gen_imgs)
            g_loss2 = self.d_loss_fn(self.discriminator(self.gen_imgs, vars=d_weights), valid)
            self.g_loss = g_loss1 + 0.1 * g_loss2
            g_grad = torch.autograd.grad(self.g_loss, self.model.module.parameters(), create_graph=self.second_order)
            g_weights = list(map(lambda p: p[1] - self.g_lr_in * p[0], zip(g_grad, self.model.module.parameters())))
            for j in range(self.train_g_times - 1):
                self.gen_imgs = self.model(inputa, vars=g_weights)
                g_loss1 = self.g_loss_fn(labela, self.gen_imgs)
                g_loss2 = self.d_loss_fn(self.discriminator(self.gen_imgs, vars=d_weights), valid)
                self.g_loss = g_loss1 + 0.1 * g_loss2
                g_grad = torch.autograd.grad(self.g_loss, g_weights, create_graph=self.second_order)
                g_weights = list(map(lambda p: p[1] - self.g_lr_in * p[0], zip(g_grad, g_weights)))
            # 1st inner loss for generator
            self.gen_imgs = self.model(inputb, vars=g_weights)
            g_loss1 = self.g_loss_fn(labelb, self.gen_imgs)
            g_loss2 = self.d_loss_fn(self.discriminator(self.gen_imgs, vars=d_weights), valid1)
            self.g_loss = g_loss1 + 0.1 * g_loss2
            g_task_lossesb.append(self.g_loss)

            for jj in range(self.task_iter - 1):
                # -------------------
                # Train discriminator
                # -------------------
                d_real_loss = self.d_loss_fn(self.discriminator(labela, vars=d_weights), valid)
                self.gen_imgs = self.model(inputa, vars=g_weights)
                d_fake_loss = self.d_loss_fn(self.discriminator(self.gen_imgs.detach(), vars=d_weights), fake)
                self.d_loss = (d_real_loss + d_fake_loss) / 2
                d_grad = torch.autograd.grad(self.d_loss, d_weights, create_graph=self.second_order)
                d_weights = list(map(lambda p: p[1] - self.d_lr_in * p[0], zip(d_grad, d_weights)))
                for j in range(self.train_d_times - 1):
                    d_real_loss = self.d_loss_fn(self.discriminator(labela, vars=d_weights), valid)
                    self.gen_imgs = self.model(inputa, vars=g_weights)
                    d_fake_loss = self.d_loss_fn(self.discriminator(self.gen_imgs.detach(), vars=d_weights), fake)
                    self.d_loss = (d_real_loss + d_fake_loss) / 2
                    d_grad = torch.autograd.grad(self.d_loss, d_weights, create_graph=self.second_order)
                    d_weights = list(map(lambda p: p[1] - self.d_lr_in * p[0], zip(d_grad, d_weights)))
                # inner losses for discriminator
                d_real_loss = self.d_loss_fn(self.discriminator(labelb, vars=d_weights), valid1)
                self.gen_imgs = self.model(inputb, vars=g_weights)
                d_fake_loss = self.d_loss_fn(self.discriminator(self.gen_imgs.detach(), vars=d_weights), fake1)
                self.d_loss = (d_real_loss + d_fake_loss) / 2
                d_task_lossesb.append(self.d_loss)

                # ---------------
                # Train generator
                # ---------------
                self.gen_imgs = self.model(inputa, vars=g_weights)
                g_loss1 = self.g_loss_fn(labela, self.gen_imgs)
                g_loss2 = self.d_loss_fn(self.discriminator(self.gen_imgs, vars=d_weights), valid)
                self.g_loss = g_loss1 + 0.1 * g_loss2
                g_grad = torch.autograd.grad(self.g_loss, g_weights, create_graph=self.second_order)
                g_weights = list(map(lambda p: p[1] - self.g_lr_in * p[0], zip(g_grad, g_weights)))
                for j in range(self.train_g_times - 1):
                    self.gen_imgs = self.model(inputa, vars=g_weights)
                    g_loss1 = self.g_loss_fn(labela, self.gen_imgs)
                    g_loss2 = self.d_loss_fn(self.discriminator(self.gen_imgs, vars=d_weights), valid)
                    self.g_loss = g_loss1 + 0.1 * g_loss2
                    g_grad = torch.autograd.grad(self.g_loss, g_weights, create_graph=self.second_order)
                    g_weights = list(map(lambda p: p[1] - self.g_lr_in * p[0], zip(g_grad, g_weights)))
                # inner losses for generator
                self.gen_imgs = self.model(inputb, vars=g_weights)
                g_loss1 = self.g_loss_fn(labelb, self.gen_imgs)
                g_loss2 = self.d_loss_fn(self.discriminator(self.gen_imgs, vars=d_weights), valid1)
                self.g_loss = g_loss1 + 0.1 * g_loss2
                g_task_lossesb.append(self.g_loss)

            task_output = [d_task_lossesb, g_task_lossesb]

            return task_output

        self.d_total_lossesb = []
        self.g_total_lossesb = []
        self.inner_loop_pre = []
        self.inner_loop_aft = []
        inputa, labela, inputb, labelb = inf
        self.model.train()
        self.discriminator.train()
        inputa = torch.as_tensor(inputa).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)
        labela = torch.as_tensor(labela).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)
        inputb = torch.as_tensor(inputb).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)
        labelb = torch.as_tensor(labelb).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)
        valid = torch.tensor(np.ones((inputa.shape[0], inputa.shape[1], 1)), requires_grad=False)
        fake = torch.tensor(np.zeros((inputa.shape[0], inputa.shape[1], 1)), requires_grad=False)
        valid = torch.as_tensor(valid).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)
        fake = torch.as_tensor(fake).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)

        valid1 = torch.tensor(np.ones((inputb.shape[0], inputb.shape[1], 1)), requires_grad=False)
        fake1 = torch.tensor(np.zeros((inputb.shape[0], inputb.shape[1], 1)), requires_grad=False)
        valid1 = torch.as_tensor(valid1).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)
        fake1 = torch.as_tensor(fake1).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)
        # LW = self.get_loss_weights()
        for i in range(self.meta_batch_size):
            # print('task number {}'.format(i+1))
            res = task_meta_learning(inputa[i], labela[i], inputb[i], labelb[i], valid[i], fake[i], valid1[i], fake1[i])
            # self.d_total_lossesb.append(sum(np.multiply(list(LW), res[0])))
            # self.g_total_lossesb.append(sum(np.multiply(list(LW), res[1])))
            self.d_total_lossesb.append(res[0][-1])
            self.g_total_lossesb.append(res[1][-1])
            self.inner_loop_pre.append(res[1][0].item())
            self.inner_loop_aft.append(res[1][-1].item())
        g_weighted_total_lossesb = sum(self.g_total_lossesb) / self.meta_batch_size
        torch.distributed.barrier()
        self.reduced_lossg = reduce_mean(g_weighted_total_lossesb, self.nprocs)
        d_weighted_total_lossesb = sum(self.d_total_lossesb) / self.meta_batch_size
        torch.distributed.barrier()
        self.reduced_lossd = reduce_mean(d_weighted_total_lossesb, self.nprocs)

        self.d_opt.zero_grad()
        d_weighted_total_lossesb.backward()
        self.d_opt.step()

        self.g_opt.zero_grad()
        g_weighted_total_lossesb.backward()
        self.g_opt.step()

    def __call__(self):

        print('[*] Training meta SR 2D GAN')
        print('Meta batch size:', self.meta_batch_size, '\tMeta g_lr:', self.g_lr_out, '\tMeta d_lr:', self.d_lr_out,
              '\tMeta iterations:', self.meta_iter)
        print('Task batch size:', self.task_batch_size, '\tTask g_lr:', self.g_lr_in, '\tTask d_lr:', self.d_lr_in,
              '\tTask iterations:', self.task_iter)

        step = self.step
        t2 = time.time()
        gloss_pre = []
        dloss_pre = []
        while True:
            # load data
            inf = dataset_loader.make_data_tensor_sr_2d(tasks_path=self.data_path, meta_batch=self.meta_batch_size,
                                                        task_batch=self.task_batch_size, spt_num=self.spt,
                                                        flag_filter=self.filter_flag, sigma=self.filter_sigma)
            # meta training
            self.construct_model(inf)

            gloss_pre.append(self.reduced_lossg.item())
            dloss_pre.append(self.reduced_lossd.item())

            step += 1
            self.current_step = step

            if step % self.print_iter == 0:
                t1 = t2
                t2 = time.time()
                if self.local_rank == 0:
                    print('Epoch:%.6d' % step, 'Generator (Pre, Post) Loss %.6f,' % (np.mean(self.inner_loop_pre)),
                          '%.6f' % (np.mean(self.inner_loop_aft)),
                          'Time: %.2f' % (t2 - t1))

            if step % self.save_iter == 0:
                g_val_loss, d_val_loss = self.validation()
                self.g_scheduler.step(np.mean(gloss_pre))
                self.d_scheduler.step(np.mean(gloss_pre))
                print_time()
                print('Epoch:%.6d' % step, 'G_Meta_loss %.6f,' % (np.mean(gloss_pre)),
                      'D_Meta_loss %.6f' % (np.mean(dloss_pre)))
                if self.local_rank == 0:
                    self.test(step, self.test_path, self.test_save_path)
                    save_checkpoint(self.model, self.g_opt, self.checkpoint_dir, step, flag=0)
                    save_checkpoint(self.discriminator, self.d_opt, self.checkpoint_dir, step, flag=1)
                    save_model(self.model, self.checkpoint_dir, step, flag=0)
                    save_model(self.discriminator, self.checkpoint_dir, step, flag=1)
                    self.writer.add_scalar('G_loss', np.mean(gloss_pre), step)
                    self.writer.add_scalar('D_loss', np.mean(dloss_pre), step)
                    self.writer.add_scalar('G_lr', self.g_opt.state_dict()['param_groups'][0]['lr'], step)
                    self.writer.add_scalar('D_lr', self.d_opt.state_dict()['param_groups'][0]['lr'], step)
                    self.writer.add_scalar('G_val_loss', g_val_loss, step)
                    self.writer.add_scalar('D_val_loss', d_val_loss, step)
                gloss_pre = []
                dloss_pre = []

            if step == self.meta_iter:
                print('Done Training')
                print_time()
                break
        self.writer.close()

    def validation(self):
        inputa, labela, inputb, labelb = dataset_loader.make_data_tensor_sr_2d(tasks_path=self.val_path, meta_batch=1,
                                                                               task_batch=self.val_num,
                                                                               spt_num=self.spt_val,
                                                                               flag_filter=self.filter_flag,
                                                                               sigma=self.filter_sigma)
        inputa, inputb, labela, labelb = inputa[0], inputb[0], labela[0], labelb[0]
        valid = torch.tensor(np.ones((inputa.shape[0], 1)), requires_grad=False)
        fake = torch.tensor(np.zeros((inputa.shape[0], 1)), requires_grad=False)
        valid = torch.as_tensor(valid).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)
        fake = torch.as_tensor(fake).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)

        inputa = torch.as_tensor(inputa).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)
        labela = torch.as_tensor(labela).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)

        # -------------------
        # Update discriminator
        # -------------------
        d_real_loss = self.d_loss_fn(self.discriminator(labela, vars=None), valid)
        gen_imgs = self.model(inputa, vars=None)
        d_fake_loss = self.d_loss_fn(self.discriminator(gen_imgs.detach(), vars=None), fake)
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_grad = torch.autograd.grad(d_loss, self.discriminator.module.parameters(), create_graph=self.second_order)
        d_weights = list(map(lambda p: p[1] - self.d_lr_in * p[0], zip(d_grad, self.discriminator.module.parameters())))
        for j in range(self.train_d_times - 1):
            d_real_loss = self.d_loss_fn(self.discriminator(labela, vars=d_weights), valid)
            gen_imgs = self.model(inputa, vars=None)
            d_fake_loss = self.d_loss_fn(self.discriminator(gen_imgs.detach(), vars=d_weights), fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_grad = torch.autograd.grad(d_loss, d_weights, create_graph=self.second_order)
            d_weights = list(map(lambda p: p[1] - self.d_lr_in * p[0], zip(d_grad, d_weights)))

        # ---------------
        # Update generator
        # ---------------
        gen_imgs = self.model(inputa, vars=None)
        g_loss1 = self.g_loss_fn(labela, gen_imgs)
        g_loss2 = self.d_loss_fn(self.discriminator(gen_imgs, vars=d_weights), valid)
        g_loss = g_loss1 + 0.1 * g_loss2
        g_grad = torch.autograd.grad(g_loss, self.model.module.parameters(), create_graph=self.second_order)
        g_weights = list(map(lambda p: p[1] - self.g_lr_in * p[0], zip(g_grad, self.model.module.parameters())))
        for j in range(self.train_g_times - 1):
            gen_imgs = self.model(inputa, vars=g_weights)
            g_loss1 = self.g_loss_fn(labela, gen_imgs)
            g_loss2 = self.d_loss_fn(self.discriminator(gen_imgs, vars=d_weights), valid)
            g_loss = g_loss1 + 0.1 * g_loss2
            g_grad = torch.autograd.grad(g_loss, g_weights, create_graph=self.second_order)
            g_weights = list(map(lambda p: p[1] - self.g_lr_in * p[0], zip(g_grad, g_weights)))

        for jj in range(self.task_iter * 2 - 1):
            # -------------------
            # Update discriminator
            # -------------------
            d_real_loss = self.d_loss_fn(self.discriminator(labela, vars=d_weights), valid)
            gen_imgs = self.model(inputa, vars=g_weights)
            d_fake_loss = self.d_loss_fn(self.discriminator(gen_imgs.detach(), vars=d_weights), fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_grad = torch.autograd.grad(d_loss, d_weights, create_graph=self.second_order)
            d_weights = list(map(lambda p: p[1] - self.d_lr_in * p[0], zip(d_grad, d_weights)))
            for j in range(self.train_d_times - 1):
                d_real_loss = self.d_loss_fn(self.discriminator(labela, vars=d_weights), valid)
                gen_imgs = self.model(inputa, vars=g_weights)
                d_fake_loss = self.d_loss_fn(self.discriminator(gen_imgs.detach(), vars=d_weights), fake)
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_grad = torch.autograd.grad(d_loss, d_weights, create_graph=self.second_order)
                d_weights = list(map(lambda p: p[1] - self.d_lr_in * p[0], zip(d_grad, d_weights)))
            # ---------------
            # Update generator
            # ---------------
            gen_imgs = self.model(inputa, vars=g_weights)
            g_loss1 = self.g_loss_fn(labela, gen_imgs)
            g_loss2 = self.d_loss_fn(self.discriminator(gen_imgs, vars=d_weights), valid)
            g_loss = g_loss1 + 0.1 * g_loss2
            g_grad = torch.autograd.grad(g_loss, g_weights, create_graph=self.second_order)
            g_weights = list(map(lambda p: p[1] - self.g_lr_in * p[0], zip(g_grad, g_weights)))
            for j in range(self.train_g_times - 1):
                gen_imgs = self.model(inputa, vars=g_weights)
                g_loss1 = self.g_loss_fn(labela, gen_imgs)
                g_loss2 = self.d_loss_fn(self.discriminator(gen_imgs, vars=d_weights), valid)
                g_loss = g_loss1 + 0.1 * g_loss2
                g_grad = torch.autograd.grad(g_loss, g_weights, create_graph=self.second_order)
                g_weights = list(map(lambda p: p[1] - self.g_lr_in * p[0], zip(g_grad, g_weights)))

        g_val_loss1 = 0
        d_val_loss1 = 0
        with torch.no_grad():
            for jj in range(inputb.shape[0]):
                inputb1 = inputb[jj, ...]
                inputb1 = inputb1[np.newaxis, ...]
                labelb1 = labelb[jj, ...]
                labelb1 = labelb1[np.newaxis, ...]
                inputb1 = torch.as_tensor(inputb1).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)
                labelb1 = torch.as_tensor(labelb1).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)
                valid = torch.tensor(np.ones((inputb1.shape[0], 1)), requires_grad=False)
                fake = torch.tensor(np.zeros((inputb1.shape[0], 1)), requires_grad=False)
                valid = torch.as_tensor(valid).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)
                fake = torch.as_tensor(fake).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)
                d_real_loss = self.d_loss_fn(self.discriminator(labelb1, vars=d_weights), valid)
                gen_imgs = self.model(inputb1, vars=g_weights)
                d_fake_loss = self.d_loss_fn(self.discriminator(gen_imgs.detach(), vars=d_weights), fake)
                d_val_loss = (d_real_loss + d_fake_loss) / 2
                torch.distributed.barrier()
                dreduced_loss = reduce_mean(d_val_loss, self.nprocs)
                d_val_loss1 += dreduced_loss.item()
                g_loss1 = self.g_loss_fn(labelb1, gen_imgs)
                g_loss2 = self.d_loss_fn(self.discriminator(gen_imgs, vars=d_weights), valid)
                g_val_loss = g_loss1 + 0.1 * g_loss2
                torch.distributed.barrier()
                greduce_loss = reduce_mean(g_val_loss, self.nprocs)
                g_val_loss1 += greduce_loss.item()
        return g_val_loss1, d_val_loss1

    def test(self, iter, test_path, save_path):
        fs = os.listdir(test_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model.eval()
        with torch.no_grad():
            for i in range(len(fs)):
                inputs = tiff.imread(test_path + '/' + fs[i])
                if len(inputs.shape) < 3:
                    inputs = inputs[np.newaxis, ...]
                inputs = inputs[np.newaxis, ...]
                inputs = inputs / 65535
                inputs = torch.as_tensor(inputs).type(torch.FloatTensor).cuda(self.local_rank, non_blocking=True)
                outs = self.model(inputs)
                outs = outs.cpu().numpy()
                outs = np.squeeze(outs)
                outs = np.uint16(XxPrctileNorm(outs) * 10000)
                tiff.imwrite(save_path + '/' + '%02d' % (i + 1) + '_iter' + '%06d' % iter + '.tif', outs)


if __name__ == '__main__':
    main(nprocs=nprocs, args=args)
