"""
the code for training RL-DFN model
Created on Tue Sep 30 19:31:36 2023
@ Last Updated by Lin Yuhuan
"""
import glob
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random
import time
from models.Dual_Cycle_section_rldeconv_train import DualGANASectionModel
from utils.dataloader import Data_File_MRC
from torch.utils.data import DataLoader
from utils.read_mrc import read_mrc
import os
from math import pi
from torch.utils.tensorboard import SummaryWriter
import argparse
import tqdm

def parse_option():
    parser = argparse.ArgumentParser(
        'RL-DFN training and evaluation script', add_help=False)

    # easy config modification
    # path to data for training
    parser.add_argument('--Data_path', type=str, default="/Code_for_3D_IsoRecon/data/TrainData", help='path to dataset')
    # path to log,checkpoints and visualization save
    parser.add_argument('--Save_path', type=str, default="/Code_for_3D_IsoRecon/train_RL_DFN", help='path to log')
    #The structure want to train
    parser.add_argument('--Struct', type=int, default=0)

    # The model's name
    parser.add_argument('--name', type=str, default="test")


    # The GPU device id you want to use
    parser.add_argument('--gpuid', type=int, default=0)
    #train process setting
    parser.add_argument('--epoches', type=int,default=10)
    parser.add_argument('--sampletimes', type=int, default=1000)
    parser.add_argument('--n_epochs_decay', type=int, default=100)
    parser.add_argument('--display_freq', type=int, default=100)
    # tilt angle of objective
    parser.add_argument('--Rotate', type=float, default=30.8, help='tilt angle of objective')
    # the crop size when training
    parser.add_argument('--patch_x_size', type=int, default=256)
    parser.add_argument('--patch_y_size', type=int, default=256)
    parser.add_argument('--patch_z_size', type=int, default=64)
    # the size of whole image
    parser.add_argument('--IMAGE_H', type=int, default=768)
    parser.add_argument('--IMAGE_W', type=int, default=768)
    parser.add_argument('--IMAGE_Z', type=int, default=100)
    # train batch size
    parser.add_argument('--batch_size', type=int,  default=1, help="batch size for train")
    # path to psfA/psfB
    parser.add_argument('--psfA_root', type=str, default="./Code_for_3D_IsoRecon/PSFs/PSF_A1.tif", help='path to psfA')
    parser.add_argument('--psfB_root', type=str, default="./Code_for_3D_IsoRecon/PSFs/PSF_B1.tif", help='path to psfB')
    #the pretrained model you want to load
    parser.add_argument('--load_checkpoint', default=None, help='checkpoint want to load')
    # the path to mrc head
    parser.add_argument('--mrc_root', default="./Code_for_3D_IsoRecon/utils/mrc/test.mrc")
    # learning rate
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--optimizer", type=str, default="adamw")

    # Section get function setting
    parser.add_argument("--Section_rotate", type=float, default=30)
    parser.add_argument("--Section_Update_Frequent", type=int, default=20)
    parser.add_argument("--Save_Epoch_Frequent", type=int, default=1000)
    parser.add_argument("--Save_Frequent", type=int, default=1000)
    parser.add_argument("--Print_Frequent", type=int, default=1000)
    # weight of loss
    parser.add_argument("--TV_loss_weight", type=float, default=0.001)
    parser.add_argument("--Cycle_loss_weight", type=float, default=50)

    parser.add_argument("--visualization", type=bool, default=True)

    args, unparsed = parser.parse_known_args()

    return args


if __name__ == '__main__':
    # Prepare data path
    Source_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = parse_option()
    name = args.name   #model name
    gpu_ids = [args.gpuid]   #GPU id
    visualization = args.visualization #if need visualization
    device = "cuda:" + str(gpu_ids[0])

    # set loss weight
    TV_loss_weight = args.TV_loss_weight
    cycle_loss_weight = args.Cycle_loss_weight

    # training parameters
    print_freq = args.Print_Frequent
    save_latest_freq = args.Save_Frequent
    save_by_iter = "True"
    save_epoch_freq = args.Save_Epoch_Frequent
    n_epochs = args.epoches
    n_epochs_decay =  args.n_epochs_decay
    display_freq = args.display_freq
    batch_size = args.batch_size
    pixel_off_update = args.Section_Update_Frequent #section theta uptade prequent

    #crop size used during training
    patch_x_size = args.patch_x_size
    patch_y_size = args.patch_y_size
    patch_z_size = args.patch_z_size
    H =  args.IMAGE_H
    W =  args.IMAGE_W
    Z =  args.IMAGE_Z

    #Save the node and visualize the node path
    checkpoints_root = Source_path +args.Save_path +"/" +name+ "/checkpoints"
    visualization_root = Source_path +args.Save_path+"/" +name+ "/visualization"
    log_root = Source_path + args.Save_path+"/" +name+ "/logs"

    #List of image file names for perspective A, perspective B...
    root = Source_path+args.Data_path + "/*"
    file_list = glob.glob(root)
    rot_view_1 = []
    rot_view_2 = []
    unrot_view = []

    #prepare training data
    for struct in [args.Struct]:
        structroot = file_list[struct]
        viewAroot = structroot + "/ViewA/*.tif"
        viewBroot = structroot + "/ViewB/*.tif"
        rot_view_1 += glob.glob(viewAroot)
        rot_view_2 += glob.glob(viewBroot)
        unrot_view += glob.glob(viewBroot)


    rot_view_1 = sorted(rot_view_1)
    rot_view_2 = sorted(rot_view_2)
    unrot_view  = sorted(unrot_view)

    head,_ = read_mrc(args.mrc_root)

    file_length = len(rot_view_1)

    #Data set construction, sample_nums is the number of sample crop on each large graph,
    # TIF False is read according to.mrc, otherwise TIF is read according to thred_hold is used to determine whether there is a structure region,
    # the image value is reduced to 0-1, if greater than the threshold value, structure is considered
    x_range = [patch_x_size//2+1, H - (patch_x_size//2+1)]
    y_range = [patch_y_size//2+1, W - (patch_y_size//2+1)]
    z_range = [patch_z_size//2+1, Z - (patch_z_size//2+1)]
    #z_range = [70, 90]
    dataset = Data_File_MRC(rot_view_1=rot_view_1,rot_view_2= rot_view_2, unrot_view=unrot_view,patch_x=patch_x_size, patch_y=patch_y_size,
                            patch_z=patch_z_size,sample_nums=args.sampletimes, TIF=True,x_range=x_range,y_range=y_range,z_range=z_range,pad=True)

    # prepare dataloader
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=1, drop_last=True, shuffle=False)

    # Save path created
    if not os.path.exists(visualization_root):
        os.mkdir(visualization_root)
    if not os.path.exists(checkpoints_root):
        os.mkdir(checkpoints_root)
    if not os.path.exists(log_root):
        os.mkdir(log_root)

    #tensorboard created
    writer = SummaryWriter(log_root)


    #model create
    model = DualGANASectionModel(patch_x=patch_x_size, patch_y=patch_y_size, patch_z=patch_z_size,name=name, gpu_ids=gpu_ids, input_nc=1,
                                 psfA_root=args.psfA_root,psfB_root=args.psfB_root,TV_loss_weight=TV_loss_weight,
                                 device=device,batch_size=batch_size, cycle_loss_weight=cycle_loss_weight,lr=args.lr,beta1=args.beta1)

    # Sets the offset of the transverse translation of the section
    pixel_off = random.uniform(0, 1)
    #Set the dynamic Angle rotation range
    theta_off = random.uniform(-pi * (args.Section_rotate / 180), pi * (args.Section_rotate / 180))

    #The double interpolation weight of the section plane is prepared in advance.
    # H and W are the length and width of the section plane, which should be specified according to the crop size during training and the incidence Angle of the data
    H = patch_z_size-6
    W = patch_y_size - 8
    theta = args.Rotate
    model.section_prepare(H=H, W=W, head=head, pixel_off=pixel_off,
                          theta_off=0,theta=theta)
    model.setup()


    total_iters = 0
    load_checkpoint = args.load_checkpoint

    #Load the pre-trained model
    if load_checkpoint:
        model.load_checkpoint(load_checkpoint)

    print("Model hyperparameters documented on tensorboard.")
    print("start the epoch training...")

    #start training
    count = 0
    for epoch in range(n_epochs + 1):
        epoch_iter = 0


        i=-1
        for  data in tqdm.tqdm(train_loader):
            i+=1
            epoch_iter+=1
            model.set_input(data)
            model.optimize_parameters()
            model.total_iters = total_iters
            if visualization:
                model.visualization(visualization_root)
            losses = model.get_current_losses()
            D_A_loss = losses['D_A']
            G_A_loss = losses['G_A']
            G_B_loss = losses['G_B']
            cycle1_loss = losses['cycle1']
            cycle2_loss = losses['cycle2']
            D_B_loss = losses['D_B']
            writer.add_scalar('D_A', D_A_loss, global_step=count, walltime=None)
            writer.add_scalar('D_B', D_B_loss, global_step=count, walltime=None)
            writer.add_scalar('G_A', G_A_loss, global_step=count, walltime=None)
            writer.add_scalar('G_B', G_B_loss, global_step=count, walltime=None)
            writer.add_scalar('cycle1', cycle1_loss, global_step=count, walltime=None)
            writer.add_scalar('cycle2', cycle2_loss, global_step=count, walltime=None)
            count += 1

            if i % pixel_off_update == 0:
                pixel_off = random.uniform(0, 1)
                theta_off = random.uniform(-pi * (30 / 180), pi * (30 / 180))
                model.section_prepare(H=60, W=120, head=head, pixel_off=pixel_off,
                                      theta_off=theta_off)

            if i % print_freq == 0:  # print training losses and save logging information to the disk
                print("----------------------------------")
                print("exp name: " + str(name) + ", gpu_id:" + str(gpu_ids))
                print("----------------------------------")
                losses = model.get_current_losses()
                print("current loss:", losses)

            if total_iters % save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                ckp_root = checkpoints_root + "/iteration" + str(total_iters) + "/"
                epoch_progress = round(float(epoch_iter / dataset_size), 2) * 100
                print('saving the latest model (epoch %d, epoch_progress %d%%)' % (epoch, epoch_progress))
                if not os.path.exists(ckp_root):
                    os.mkdir(ckp_root)

                model.save_checkpoint(ckp_root)

            total_iters += 1
            iter_data_time = time.time()

        # display the image histogram per epoch.
        losses = model.get_current_losses()
        print("epoch i loss:", losses)
        model.update_learning_rate()
