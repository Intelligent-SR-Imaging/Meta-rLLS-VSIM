"""
Step2: Reconstructed into a laterally isotropic SR volume using the Meta-VSI-SR model
Created on Tue Sep 30 19:31:36 2023
@ Last Updated by Lin Yuhuan
"""
import numpy as np
import os
import sys

Source_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(Source_path)
os.chmod(Source_path, 0o755)

import torch
from Code_for_2D_IsoRecon.model.meta_SR_2D_GAN import Generator
from Code_for_2D_IsoRecon.utils.fun_recon import isotropic_recon, rotation
import argparse

def parse_option():
    parser = argparse.ArgumentParser()
    # data_path has LLSM data with all timepoints (001.tif, 002.tif, ...)
    parser.add_argument("--Data_path", type=str, default='/Demo_Data_for_3D_IsoRecon/step1/Factin/488')
    # save_path for Iso-XY reconstruction
    parser.add_argument("--Save_path", type=str, default='/Demo_Data_for_3D_IsoRecon/step2')
    # model_path has a fine-tuned model of a specific structure for Iso-XY reconstruction
    parser.add_argument("--model_path", type=str, default='/Code_for_2D_IsoRecon/finetuned_model/Factin/F-actin.pth')

    # GPU device number
    parser.add_argument("--cuda_num", type=int, default=0)
    # otf path for Iso-XY reconstruction
    parser.add_argument("--otf_path", type=str, default='/Code_for_2D_IsoRecon/OTF/')

    # save_name of image
    parser.add_argument("--save_name", type=str, default='F-actin')
    # excitation wavelength of the LLSM data
    parser.add_argument("--wavelength", type=float, default=488e-3)
    # weight factor of otf for reconstruction
    parser.add_argument("--otf_weight", type=float, default=8.0)
    # wiener factor for reconstruction
    parser.add_argument("--wiener_weight", type=float, default=0.1)
    # flag of adjusting intensity among slices
    parser.add_argument("--adjust_intens", type=int, default=1)
    # threshold of adjusting intensity
    parser.add_argument("--adjust_thresh", type=float, default=99.0)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    Source_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = parse_option()

    # data_path has LLSM data with all timepoints (001.tif, 002.tif, ...)
    data_path = Source_path + args.Data_path
    files = np.sort(os.listdir(data_path))

    # loading single direction SR model
    model_path = Source_path + args.model_path
    generator = Generator(in_channel=7, n_ResGroup=4, n_RCAB=4, out_channel=3)
    generator.load_state_dict(torch.load(model_path))
    print('loading meta model:', model_path)
    torch.cuda.set_device(args.cuda_num)
    device = torch.device('cuda')

    # save path of raw data and single direction SR data with 3 orientations (0, 60, 120 degrees)
    WF_rotation_path = Source_path + args.Save_path + '/origin_rotation'
    SR_rotation_path = Source_path + args.Save_path + '/sr_rotation'
    save_path1 = WF_rotation_path
    save_path2 = SR_rotation_path

    # infer and reconstruct LLSM stacks timepoints by timepoints
    tp = 1
    tp1 = 0
    otf_path = Source_path + args.otf_path
    # IsoXY reconstruction: using data from otf_path, save_path1 and save_path2 to reconstruct, outputs saved in save_path5.
    save_path = Source_path + args.Save_path + '/recon'
    for fs in files:
        fs = data_path + '/' + fs
        print(fs)
        # infer single direction SR images with 3 orientations
        Nz = rotation(fs, generator, device, save_path1, save_path2, tp)
        tp1 += 1

        # --------------------------------------------------
        # save path of reconstruction images
        save_path5 = save_path + '/' + '%.4d/' % tp1
        if not os.path.exists(save_path5):
            os.makedirs(save_path5)

        # 488, 560, 642
        # if NZ of LLSM-WF is 181, then here Nz = 181-6; tp=Timepoints; lambda: 488 or 560 or 642; core_num: multi-cpu processing for tp files
        # needn't change other params
        param = {'Nz': Nz - 6, 'tp': tp, 'lambda': args.wavelength, 'core_num': 1,
                 'n_ang': 3, 'wf_sigma': args.wiener_weight, 'save_name': args.save_name,
                 'offset_ang': 0, 'apo_factor': 2, 'isApoBand0': True, 'image_folder': save_path1, 'wf': save_path2,
                 'otf_weight': args.otf_weight, 'use_origin_sr0': False, 'origin_sr0': '', 'z_offset': 0,
                 'z_range': 0, 'adjust_intens': args.adjust_intens, 'adjust_thresh': args.adjust_thresh}
        # reconstruction
        isotropic_recon(otf_path, save_path5, param)

