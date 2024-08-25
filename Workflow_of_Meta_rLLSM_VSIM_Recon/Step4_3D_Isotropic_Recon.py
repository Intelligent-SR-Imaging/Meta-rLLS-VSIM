"""
Step4: use pretrained RL-DFN to reconstruct a whole 3D Image
Created on Tue Sep 30 19:31:36 2023
@ Last Updated by Lin Yuhuan
"""
import os
import tifffile
import tqdm
import sys
Source_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(Source_path)
os.chmod(Source_path, 0o755)

import torch
from Code_for_3D_IsoRecon.models.Dual_Cycle_section_rldeconv_test import DualGANASectionModel
from Code_for_3D_IsoRecon.utils.read_mrc import read_mrc
from Code_for_3D_IsoRecon.utils.utils import Align,padding
import numpy as np
import glob
import argparse

def parse_option():
    parser = argparse.ArgumentParser(
        'RL-DFN inference script', add_help=False)
    # path to data for testing
    parser.add_argument('--Data_path', type=str, default='/Demo_Data_for_3D_IsoRecon/step3', help='path to dataset')
    # path to save
    parser.add_argument('--Save_path', type=str, default='/Demo_Data_for_3D_IsoRecon/step4', help='path to dataset')
    # path to pretrained model
    parser.add_argument('--model_path', default='/Code_for_3D_IsoRecon/finetuned_model/F-actin', help='checkpoint want to load')
    # The model's name
    parser.add_argument('--name', type=str, default="F-actin")
    # The GPU device id you want to use
    parser.add_argument('--gpuid', type=int, default=0)
    # tilt angle of objective
    parser.add_argument('--Rotate', type=float, default=30.8, help='tilt angle of objective')
    # the crop size when training
    parser.add_argument('--patch_x_size', type=int, default=256)
    parser.add_argument('--patch_y_size', type=int, default=256)
    parser.add_argument('--patch_z_size', type=int, default=108)
    # the size of whole image
    parser.add_argument('--IMAGE_H', type=int, default=768)
    parser.add_argument('--IMAGE_W', type=int, default=768)
    parser.add_argument('--IMAGE_Z', type=int, default=101)


    # path to psfA/psfB
    parser.add_argument('--psfA_root', type=str, default="/Code_for_3D_IsoRecon/PSFs/PSF_A1.tif", help='path to psfA')
    parser.add_argument('--psfB_root', type=str, default="/Code_for_3D_IsoRecon/PSFs/PSF_B1.tif", help='path to psfB')
    # the path to mrc head
    parser.add_argument('--mrc_root', default="/Code_for_3D_IsoRecon/utils/mrc/test.mrc")

    args, unparsed = parser.parse_known_args()

    return args

if __name__ == "__main__":

    #test data prepare
    Source_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = parse_option()
    root = Source_path + args.Data_path+"/*.tif"
    save_path = Source_path + args.Save_path+"/"+args.name+"/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_list = glob.glob(root)
    rot_view_1 = []
    rot_view_2 = []
    unrot_view = []

    for name_ in file_list:

        if "A" in name_:
            rot_view_1.append(name_)
        elif "B" in name_:
            rot_view_2.append(name_)
        #else:
            unrot_view.append(name_)

    rot_view_1 = sorted(rot_view_1)
    rot_view_2 = sorted(rot_view_2)
    unrot_view = sorted(unrot_view)

    #set experiment name and checkpoint save path
    name = args.name
    checkpoints_root = Source_path + args.model_path+"/"
    #decide the cuda you want to use
    gpu_ids = [args.gpuid]
    device = "cuda:"+str(args.gpuid)

    #In the paper, we use Unet structure. Prepare the mrc head
    model_type = "Unet"
    psf_know = True
    mrc_root = Source_path + args.mrc_root
    head,_ = read_mrc(mrc_root)

    #prepare the crop size before test
    patch_x_size = args.patch_x_size+16
    patch_y_size = args.patch_y_size+16
    patch_z_size = args.patch_z_size
    x_size = args.patch_x_size
    y_size = args.patch_y_size
    z_size = args.patch_z_size
    x_steps = args.IMAGE_H // x_size
    y_steps = args.IMAGE_W // y_size

    patch_x_re_ = args.IMAGE_H - x_steps * x_size
    patch_y_re_ = args.IMAGE_W - y_steps * y_size
    patch_x_re = patch_x_re_ + 16
    patch_y_re = patch_y_re_ + 16

    psfA_root = Source_path + args.psfA_root
    psfB_root = Source_path + args.psfB_root
    #creat a RL-DFN model
    model = DualGANASectionModel(patch_x=patch_x_size, patch_y=patch_y_size, patch_z=patch_z_size, name=name,
                                 gpu_ids=gpu_ids, input_nc=1,
                                 psfA_root=psfA_root, psfB_root=psfB_root,
                                 device=device,patch_x_re=patch_x_re,patch_y_re=patch_y_re)
    # prepare the section function
    H = patch_z_size - 6
    W = patch_y_size - 8
    theta = args.Rotate
    model.section_prepare(H=H, W=W, head=head, pixel_off=0,
                          theta_off=0, theta=theta)

    model.load_checkpoint(checkpoints_root)
    print("load complete")
    model.set_eval()

    file_len = len(rot_view_1)



    #deal all pairs in data
    for index in tqdm.tqdm(range(file_len)):


        view1_ = tifffile.imread(rot_view_1[index]).transpose(2, 1, 0).astype(np.float32)
        view2_ = tifffile.imread(rot_view_2[index]).transpose(2, 1, 0).astype(np.float32)
        z_off = args.patch_z_size-view1_.shape[2]
        print("Z direction padding size:", z_off)



        if z_off>=0:
            pad = view1_[:, :, :z_off] * 0
            view1_ = np.concatenate((view1_, pad), axis=-1)
            view2_ = np.concatenate((view2_, pad), axis=-1)
        else:
            view1_ = view1_[:,:,  :z_off]
            view2_ = view2_[:, :, :z_off]

        view1_ = padding(view1_)
        view2_ = padding(view2_)


        xx = []
        for i in range(x_steps):
            yy = []
            for j in range(y_steps):
                print("Processing patch: x id ",i,", y id ",j)

                x_step = x_size
                y_step = y_size
                if_x = 0
                if_y = 0

                if i == x_step-1:
                    x_step = patch_x_re_
                    if_x = 1

                if j == y_step-1:
                    y_step = patch_y_re_
                    if_y = 1

                model.update_crop_size(x_step + 16, y_step + 16, patch_z_size, if_x, if_y)
                view1 = view1_.astype(np.float32)[x_size * i:x_size * (i) + x_step + 16, y_size * j: y_size * (j) + y_step + 16,
                        :patch_z_size]
                view2 = view2_.astype(np.float32)[x_size * i:x_size * (i) + x_step + 16, y_size * j:y_size * (j) + y_step + 16,
                        :patch_z_size]

                fusion = tifffile.imread(unrot_view[0]).transpose(2, 1, 0).astype(np.float32)


                view1 = torch.from_numpy(view1)
                view2 = torch.from_numpy(view2)
                fusion = torch.from_numpy(fusion)

                view1 = view1 / view1.max()
                view2 = view2 / view2.max()
                fusion = fusion / fusion.max()


                view1 = view1.permute(2, 1, 0).unsqueeze(dim=0).unsqueeze(dim=0)
                view2 = view2.permute(2, 1, 0).unsqueeze(dim=0).unsqueeze(dim=0)
                fusion = fusion.permute(2, 1, 0).unsqueeze(dim=0).unsqueeze(dim=0)

                input = []
                input.append(view1)
                input.append(view2)
                input.append(fusion)

                model.set_input(input)

                with torch.no_grad():
                    model.forward()

                fake = model.fake

                fake = fake[:, :, :, 8:-8, 8:-8].squeeze(dim=0).squeeze(dim=0).permute(2, 1, 0)
                fake = torch.where(torch.isnan(fake), torch.full_like(fake, 0), fake)
                fake = fake.cpu().detach().numpy()



                yy.append(fake)

            xx.append(np.concatenate(yy, axis=1))

        fake = np.concatenate(xx, axis=0)
        #fake = fake / fake.max()
       # fake[fake<0] = 0
        #fake = fake * ((2 ** 16)-1)
        #fake = fake.astype(np.uint16)
        fake_root = save_path + str(index+1).zfill(3) + ".tif"
        tifffile.imwrite(fake_root, fake.transpose(2, 1, 0))



















