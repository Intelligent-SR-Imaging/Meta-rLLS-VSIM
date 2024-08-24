"""
fine-tune a VSI-SR model
Created on Tue Sep 30 19:31:36 2023
@ Last Updated by Lin Yuhuan
"""

import os
import sys

Source_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(Source_path)

import datetime
import time
from scipy.ndimage import gaussian_filter
from model.meta_SR_2D_GAN import Generator
import torch
import numpy as np
from Code_for_2D_IsoRecon.utils import utils
from Code_for_2D_IsoRecon.utils import dataset_loader

import tifffile as tiff
from torch.utils.tensorboard import SummaryWriter
import argparse


parser = argparse.ArgumentParser()
# about data for finetuning and testing
# path to fine-tune data and testing data
parser.add_argument("--Data_path", type=str, default='/Code_for_2D_IsoRecon/data/finetune')
# apply gaussian filters to GT training images to suppress artifacts from SIM reconstruction
parser.add_argument("--gt_filter_flag", type=bool, default=True)
parser.add_argument("--gt_filter_sigma", type=float, default=0.6)
# threshold of GT masks for PSNR calculation
parser.add_argument("--mask_thresh", type=float, default=0.01)

# about save path
parser.add_argument("--Save_path", type=str, default='/Code_for_2D_IsoRecon/meta_rcan/finetune')
parser.add_argument("--save_step", type=int, default=5)

# about training: iteration, batch size and printing step
parser.add_argument("--iter", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--print_step", type=int, default=2)

# about device: GPU device
parser.add_argument("--cuda_num", type=int, default=0)

# about model: meta model path used for fine-tuning
parser.add_argument("--model_path", type=str, default='/Code_for_2D_IsoRecon/trained_meta_model/meta_model.pth')
parser.add_argument("--in_channel", type=int, default=7)
parser.add_argument("--out_channel", type=int, default=3)
parser.add_argument("--num_resgroup", type=int, default=4)
parser.add_argument("--num_resblock", type=int, default=4)

# about loss fn: weights of SSIM loss and FFT loss
parser.add_argument("--ssim_param", type=float, default=0.1)
parser.add_argument("--fft_param", type=float, default=1.0)

# about lr and optim: learning rate for fine-tuning
parser.add_argument("--g_lr", type=float, default=25*1e-2)
# optimizer for fine-tuning: Adam or SGD
parser.add_argument("--optim_Adam", type=bool, default=False)
# step decay of learning rate
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
parser.add_argument("--lr_decay_step", type=list, default=[10, 20, 30])

args = parser.parse_args()
Source_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Test(object):
    def __init__(self, args):
        """Fine-tune a meta-VSI-SR model"""
        self.model_path = Source_path + args.model_path
        self.finetune_train_input_path = Source_path + args.Data_path + "/train/input"
        self.finetune_train_gt_path = Source_path + args.Data_path + "/train/gt"
        self.finetune_test_input_path = Source_path + args.Data_path + "/test/input"
        self.finetune_test_gt_path = Source_path + args.Data_path +"/test/gt"
        self.gt_filter_flag = args.gt_filter_flag
        self.gt_filter_sigma = args.gt_filter_sigma
        self.mask_thresh = args.mask_thresh
        self.save_path = Source_path + args.Save_path
        self.save_model_path = os.path.join(self.save_path, 'model')
        self.save_test_path = os.path.join(self.save_path, 'test')
        self.save_log_path = os.path.join(self.save_path, 'log')

        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        if not os.path.exists(self.save_test_path):
            os.makedirs(self.save_test_path)
        if not os.path.exists(self.save_log_path):
            os.makedirs(self.save_log_path)

        self.iter = args.iter
        self.step = 0
        self.batch_size = args.batch_size
        self.display_iter = args.print_step
        self.save_step = args.save_step

        torch.cuda.set_device(args.cuda_num)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator(in_channel=args.in_channel, out_channel=args.out_channel, n_ResGroup=args.num_resgroup, n_RCAB=args.num_resblock)
        self.generator = self.generator.to(self.device)
        self.generator.load_state_dict(torch.load(self.model_path, map_location=self.device))
        print('Loading meta model:', self.model_path)

        self.g_loss_fn = utils.loss_mse_ssim_fft(device=self.device, ssim_param=args.ssim_param, fft_param=args.fft_param).to(self.device)

        self.g_lr = args.g_lr

        if args.optim_Adam:
            self.g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.g_lr)
        else:
            self.g_opt = torch.optim.SGD(self.generator.parameters(), lr=self.g_lr)
        # self.g_scheduler = torch.optim.lr_scheduler.StepLR(self.g_opt, step_size=25000, gamma=0.5)
        self.g_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.g_opt, milestones=args.lr_decay_step, gamma=args.lr_decay_factor)

        self.writer = SummaryWriter(self.save_log_path)

        self.psnr_num = None

    def __call__(self):
        psnrs = []
        # test at iteration 0 (i.e. test before fine-tuning)
        fs = np.sort(os.listdir(self.finetune_test_input_path))
        for i in range(len(fs)):
            image_input0 = tiff.imread(self.finetune_test_input_path+'/'+fs[i])
            image_input0 = np.array(image_input0) / 65535
            image_input_out = np.array(image_input0[3:-3, ...] * 1e4).astype('uint16')
            tiff.imwrite(self.save_test_path + '/input' + '%.3d' % i + '.tif', image_input_out)
            if self.finetune_test_gt_path is not None:
                image_gt0 = tiff.imread(self.finetune_test_gt_path+'/'+fs[i])
                image_gt0 = np.array(image_gt0) / 65535
                if self.gt_filter_flag:
                    for i1 in range(image_gt0.shape[0]):
                        image_gt0[i1, ...] = gaussian_filter(image_gt0[i1, ...], self.gt_filter_sigma)
                image_gt_out = np.array(1e4 * image_gt0[3:-3, ...]).astype('uint16')
                tiff.imwrite(self.save_test_path + '/gt_blur'+ '%.3d' % i + '.tif', image_gt_out)
                self.test(image_input0, image_gt0, i, self.step, mask_thresh=self.mask_thresh)
                psnrs.append(self.psnr_num)
            else:
                self.test(image_input0, image_input0, i, self.step, mask_thresh=self.mask_thresh)
        if self.finetune_test_gt_path is not None:
            self.writer.add_scalar('val_psnr', np.mean(psnrs), self.step)
            psnrs = []

        # start fine-tuning
        gloss_pre, gloss_pre1 = [], []
        t2 = time.time()
        ts = datetime.datetime.now()
        finetune_data_dir = os.listdir(self.finetune_train_input_path)
        print(self.finetune_train_input_path)
        while self.step < self.iter:
            self.inputs, self.labels = dataset_loader.make_batch_finetune_data(image_path=self.finetune_train_input_path, image_dir=finetune_data_dir, gt_path=self.finetune_train_gt_path, batch=self.batch_size, flag_filter=self.gt_filter_flag, sigma=self.gt_filter_sigma)
            self.generator.train()
            self.inputs = torch.as_tensor(self.inputs).type(torch.FloatTensor).to(self.device)
            self.labels = torch.as_tensor(self.labels).type(torch.FloatTensor).to(self.device)
            # ---------------
            # Finetune generator
            # ---------------
            self.g_opt.zero_grad()
            self.gen_imgs = self.generator(self.inputs)
            self.g_loss = self.g_loss_fn(self.labels, self.gen_imgs)
            self.g_loss.backward()
            self.g_opt.step()
            self.g_scheduler.step()
            gloss_pre.append(self.g_loss.item())
            gloss_pre1.append(self.g_loss.item())
            self.step += 1
            if self.step % self.save_step == 0:
                torch.save(self.generator.state_dict(), self.save_model_path + '/g_model' + '%.4d' % self.step + '.pth')
                self.writer.add_scalar('Finetune loss', np.mean(gloss_pre), self.step)
                self.writer.add_scalar('lr', self.g_opt.state_dict()['param_groups'][0]['lr'], self.step)
                gloss_pre = []
                for i in range(len(fs)):
                    image_input0 = tiff.imread(self.finetune_test_input_path + '/' + fs[i])
                    image_input0 = np.array(image_input0) / 65535
                    if self.finetune_test_gt_path is not None:
                        image_gt0 = tiff.imread(self.finetune_test_gt_path + '/' + fs[i])
                        image_gt0 = np.array(image_gt0) / 65535
                        if self.gt_filter_flag:
                            for i1 in range(image_gt0.shape[0]):
                                image_gt0[i1, ...] = gaussian_filter(image_gt0[i1, ...], self.gt_filter_sigma)
                        self.test(image_input0, image_gt0, i, self.step, mask_thresh=self.mask_thresh)
                        psnrs.append(self.psnr_num)
                    else:
                        self.test(image_input0, image_input0, i, self.step, mask_thresh=self.mask_thresh)
                if self.finetune_test_gt_path is not None:
                    self.writer.add_scalar('val_psnr', np.mean(psnrs), self.step)
                    psnrs = []
            if self.step % self.display_iter == 0:
                t1 = t2
                t2 = time.time()
                print('Epoch:%.6d' % self.step, 'Finetuning_loss:%.6f' % (np.mean(gloss_pre1)), 'Time: %.2f' % (t2-t1))
                gloss_pre1 = []

        te = datetime.datetime.now()
        print('Finetuning time: %s' % (te-ts))

    def cal_avg(self, outs):
        outs1 = []
        for j in range(outs.shape[0]):
            if j == 0:
                outs1.append((outs[0][1] + outs[1][0]) / 2)
            elif j == outs.shape[0] - 1:
                outs1.append((outs[j][1] + outs[j - 1][2]) / 2)
            else:
                outs1.append((outs[j - 1][2] + outs[j][1] + outs[j + 1][0]) / 3)
        return np.array(outs1)

    def test(self, image_input, image_gt, fs, step, mask_thresh=0.01):
        self.generator.eval()
        sr_out = []
        with torch.no_grad():
            for i1 in range(3, image_input.shape[0]-3):
                input1 = image_input[i1-3:i1+4, ...]
                input1 = input1[np.newaxis, ...]
                input1 = torch.as_tensor(input1).type(torch.FloatTensor).to(self.device)
                output1 = self.generator(input1)
                output1 = output1.cpu().numpy()
                output1 = np.squeeze(output1)
                # if have gt, do linear transform
                if self.finetune_test_gt_path is not None:
                    output2 = []
                    for i2 in range(output1.shape[0]):
                        output1_n = utils.linear_tf(utils.XxPrctileNorm(output1[i2, ...]), image_gt[i1-1+i2, ...])
                        output2.append(output1_n)
                    output1 = np.array(output2)
                    # output1[output1 < 0] = 0
                sr_out.append(output1)
        sr_out = self.cal_avg(np.array(sr_out))
        # if have gt, calculate PSNR
        if self.finetune_test_gt_path is not None:
            mask = []
            for i in range(3, image_input.shape[0]-3):
                mask1 = utils.XxCalMask(utils.XxPrctileNorm(image_gt[i, ...]), thresh=mask_thresh)
                mask.append(mask1)
            mask = np.array(mask)
            self.psnr_num = utils.psnr_mask_v2(sr_out, image_gt[3:image_gt.shape[0]-3, ...], mask)
        sr_out = np.array(sr_out * 10000).astype('float32')
        tiff.imwrite(self.save_test_path + '/input' + '%.3d' % fs + '_iter' + '%.6d' % step + '.tif', sr_out)


Tester = Test(args)
Tester()


