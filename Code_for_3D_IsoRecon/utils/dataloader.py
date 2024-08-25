"""
Data loader for training
Created on Tue Sep 30 19:31:36 2023
@ Last Updated by Lin Yuhuan
"""
import torch
import numpy as np
from torch.utils.data import Dataset
import random
from utils import read_mrc
import tifffile

class Data_File_MRC(Dataset):
    """ RL-DFN Model, add function update_crop_size for test
        Args:
            rot_view_1 (str),rot_view_2 (str): two views' image file list.
            patch_x(int) ,patch_y(int) ,patch_z(int) : crops' size
            sample_nums(int): sample times in one image
            TIFF (bool): use TIFF file or not.
            x_range(list(int)), y_range(list(int)), z_range(list(int)): the range to crop randomly
    """
    def __init__(self, rot_view_1,rot_view_2,unrot_view,patch_x,patch_y,patch_z,sample_nums,TIF=False,pad = False,x_range=[],y_range=[],z_range=[]):
        self.tmp_index = -1
        self.tmp_volume_rot1 = None
        self.tmp_volume_rot2 = None
        self.tmp_volume_unrot = None
        self.rot_view_1 = rot_view_1
        self.rot_view_2 = rot_view_2
        self.unrot_view = unrot_view
        x_pos = None
        y_pos = None
        z_pos = None
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.z_pos = z_pos
        self.tif = TIF

        self.rot1_list = []
        self.rot2_list = []
        self.fusion_list = []

        self.file_num = len(self.rot_view_1)

        for file_index in range( self.file_num):
            if self.tif:
                rot1 = tifffile.imread(self.rot_view_1[file_index]).transpose(2, 1, 0)
                rot2 = tifffile.imread(self.rot_view_2[file_index]).transpose(2, 1, 0)
            else:
                _, rot1 = read_mrc.read_mrc(self.rot_view_1[file_index])
                _, rot2 = read_mrc.read_mrc(self.rot_view_2[file_index])
            unrot = tifffile.imread(self.unrot_view[file_index]).transpose(2, 1, 0)

            #rot1 = np.rot90(rot1, k=3, axes=(0, 1))
            #rot2 = np.rot90(rot2, k=3, axes=(0, 1))
            #unrot = np.rot90(unrot, k=3, axes=(0, 1))

            if pad:
                padding = np.zeros_like(rot1[:,:,:30])
                rot1 = np.concatenate((rot1,padding),axis=-1)
                rot2 = np.concatenate((rot2, padding), axis=-1)
                unrot = np.concatenate((unrot, padding), axis=-1)

            self.rot1_list.append(rot1)
            self.rot2_list.append(rot2)
            self.fusion_list.append(unrot)


        self.patch_x = patch_x
        self.patch_y = patch_y
        self.patch_z = patch_z
        self.sample_nums = sample_nums
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.each_num =sample_nums
        self.length = len(self.rot_view_1)*sample_nums

        if len(self.rot_view_1) != len(self.rot_view_2):
            print("Length Error!!!!!")


    def __len__(self):
        return self.length

    def augmentation(self, raw, type_index):
        if type_index==0:
            raw = raw[::-1,:,:]
        elif type_index==1:
            raw = raw[::-1, ::-1, :]
        elif type_index == 2:
            raw = raw[::-1, :, ::-1]
        elif type_index == 3:
            raw = raw[::-1, ::-1, ::-1]
        elif type_index == 4:
            raw = raw[:, ::-1, :]
        elif type_index == 5:
            raw = raw[:, ::-1, ::-1]
        elif type_index == 6:
            raw = raw[:, :, ::-1]

        return raw

    def get_structure_map(self,vol):
        vol = vol/vol.max()
        map = np.zeros_like(vol)

        map[vol>0.15] = 1
        map[:self.patch_x // 2, :, :] = 0
        map[-self.patch_x // 2:, :, :] = 0
        map[:, :self.patch_y // 2, :] = 0
        map[:, -self.patch_y // 2:, :] = 0
        map[:, :, :self.patch_z // 2 + 3] = 0
        map[:, :, -self.patch_z // 2 - 3:] = 0
        map_location = np.where(map)
        return map_location




    def __getitem__(self, item):

        index = item%self.file_num

        rot1 = self.rot1_list[index]
        rot2 = self.rot2_list[index]
        unrot = self.fusion_list[index]


        rot1 = rot1 / rot1.max()
        rot2 = rot2 / rot2.max()
        unrot = unrot / unrot.max()


        if self.x_pos ==None:
            x_min = self.x_range[0]
            x_max = self.x_range[1]
            y_min = self.y_range[0]
            y_max = self.y_range[1]

            z_min = self.z_range[0]
            z_max = self.z_range[1]



            x_pos = random.randint(x_min,  x_max)
            y_pos = random.randint(y_min, y_max)
            z_pos = random.randint(z_min,  z_max)

        else:
            x_pos = self.x_pos
            y_pos = self.y_pos
            z_pos = self.z_pos


        rot1_patch = rot1[x_pos-self.patch_x//2:x_pos+self.patch_x//2,y_pos-self.patch_y//2:y_pos+self.patch_y//2,z_pos-self.patch_z//2:z_pos+self.patch_z//2]
        rot2_patch = rot2[x_pos - self.patch_x // 2:x_pos + self.patch_x // 2,y_pos-self.patch_y//2:y_pos+self.patch_y//2,z_pos-self.patch_z//2:z_pos+self.patch_z//2]
        unrot_patch = unrot[x_pos - self.patch_x // 2:x_pos + self.patch_x // 2,y_pos-self.patch_y//2:y_pos+self.patch_y//2,z_pos-self.patch_z//2:z_pos+self.patch_z//2]

        rot1_patch = rot1_patch.astype(np.float32)
        rot2_patch = rot2_patch.astype(np.float32)
        unrot_patch = unrot_patch.astype(np.float32)

        rot1_patch = torch.from_numpy(rot1_patch)
        rot2_patch = torch.from_numpy(rot2_patch)
        unrot_patch = torch.from_numpy(unrot_patch)
        #print(rot1_patch.shape)

        rot1_patch = rot1_patch.permute(2, 1, 0).reshape(1, self.patch_z, self.patch_y, self.patch_x)
        rot2_patch = rot2_patch.permute(2, 1, 0).reshape(1, self.patch_z, self.patch_y, self.patch_x)
        unrot_patch = unrot_patch.permute(2, 1, 0).reshape(1, self.patch_z, self.patch_y, self.patch_x)


        return rot1_patch,rot2_patch,unrot_patch,rot1_patch,rot2_patch,unrot_patch


