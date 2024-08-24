"""
RL-Deconv module in RL-DFN and PSF degradation module
Created on Tue Sep 30 19:31:36 2023
@ Last Updated by Lin Yuhuan
"""

import time
import torch
import tifffile
import numpy as np
import tqdm
import torch.nn as nn
def Align(img,x1,y1,z1,padValue=0):
    """This function is used to a lign a 3D image to a specific size
           Args:
                img (numpy.array): a 3D image stack
                x1, y1, z1, (int): Size to align
                padValue(value) : the value for padding
    """
    x2,y2,z2 = img.shape

    x_m = max(x1,x2)
    y_m = max(y1, y2)
    z_m = max(z1, z2)

    img_tmp = torch.ones(x_m,y_m,z_m)*padValue

    x_o = round((x_m-x2)/2)
    y_o = round((y_m - y2) / 2)
    z_o = round((z_m - z2) / 2)

    img_tmp[x_o:x2+x_o,y_o:y2+y_o,z_o:z2+z_o] = img

    x_o = round((x_m - x1) / 2)
    y_o = round((y_m - y1) / 2)
    z_o = round((z_m - z1) / 2)

    img = img_tmp[x_o:x1+x_o, y_o:y1 + y_o, z_o:z1 + z_o]
    return img


class RL_Deconv(nn.Module):
    """RL-Deconv module in RL-DFN and PSF degradation module
       Args::
        psfA_root(str)，psfB_root(str): two views' estimated PSF root, TIFF file
        X_dim(int), Y_dim(int) ,Z_dim(int): the size of input crop
        de_size (int): the convolution kernel size when degradation
        device(str) : GPU device
        batch_size(int) : batch size
    """
    def __init__(self,psfA_root,psfB_root,X_dim,Y_dim,Z_dim,de_size = 19,eps = 10e-6,device="cuda:7",batch_size=1):
        super(RL_Deconv, self).__init__()
        self.eps = eps
        self.device = device
        self.batch_size= batch_size
        psfA =tifffile.imread(psfA_root).transpose(2,1,0).astype(np.float32)
        psfB = tifffile.imread(psfB_root).transpose(2,1,0).astype(np.float32)

        self.psfA_ = torch.from_numpy(psfA)
        self.psfB_ = torch.from_numpy(psfB)

        psfA = Align(self.psfA_,X_dim,Y_dim,Z_dim).permute(2,1,0)
        psfB = Align(self.psfB_, X_dim, Y_dim, Z_dim).permute(2,1,0)

        psfA = psfA/psfA.sum()
        psfB = psfB / psfB.sum()

        self.otfA = torch.fft.fftn(psfA)
        self.otfA = self.otfA.unsqueeze(dim=0).repeat(batch_size,1,1,1)
        self.otfB = torch.fft.fftn(psfB)
        self.otfB = self.otfB.unsqueeze(dim=0).repeat(batch_size, 1, 1, 1)

        psfA = Align(self.psfA_, de_size, de_size, de_size).permute(2, 1, 0)
        psfB = Align(self.psfB_, de_size, de_size, de_size).permute(2, 1, 0)

        psfA = psfA / psfA.sum()
        psfB = psfB / psfB.sum()

        self.psfA = psfA.reshape(1,1,de_size,de_size,de_size)
        self.psfB = psfB.reshape(1,1,de_size,de_size,de_size)

    def forward(self,viewA,viewB,fusion):
        """
        RL Deconv process
        """
        fft_fusion = torch.fft.fftn(fusion)

        blur = torch.real(torch.fft.ifftshift(torch.fft.ifftn(fft_fusion*self.otfA)))+0.00001
        curA = torch.clamp(viewA/blur,min=self.eps)
        curA = torch.real(torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftn(curA)*self.otfA)))

        blur = torch.real(torch.fft.ifftshift(torch.fft.ifftn(fft_fusion * self.otfB)))+0.00001
        curB = torch.clamp(viewB / blur, min=self.eps)
        curB = torch.real(torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftn(curB) * self.otfB)))

        return curA,curB

    def set_device(self,device):
        self.otfA = self.otfA.to(self.device)
        self.otfB = self.otfB.to(self.device)
        self.psfA = self.psfA.to(self.device)
        self.psfB = self.psfB.to(self.device)

    def degradation(self,img):
        """
        degradation by estimated PSF
        """
        viewA = nn.functional.conv3d(img,self.psfA,padding="same")
        viewB = nn.functional.conv3d(img, self.psfB, padding="same")

        return viewA,viewB

    def update_crop_size(self,patch_x,patch_y,patch_z):
        psfA = Align(self.psfA_, patch_x, patch_y, patch_z).permute(2, 1, 0)
        psfB = Align(self.psfB_, patch_x, patch_y, patch_z).permute(2, 1, 0)

        psfA = psfA / psfA.sum()
        psfB = psfB / psfB.sum()

        self.otfA = torch.fft.fftn(psfA)
        self.otfA = self.otfA.unsqueeze(dim=0).repeat(self.batch_size, 1, 1, 1)
        self.otfB = torch.fft.fftn(psfB)
        self.otfB = self.otfB.unsqueeze(dim=0).repeat(self.batch_size, 1, 1, 1)


if __name__ == "__main__":
    iter = 3
    device = "cuda:6"
    '''_,view1 = read_mrc("./ViewA_5.mrc")

    _, view2 = read_mrc("./ViewB_5.mrc")'''
    view1 = tifffile.imread("./ViewA.tif").transpose(2,1,0)
    view2 = tifffile.imread("./ViewB.tif").transpose(2,1,0)
    print(view1.shape)

    rl_deconv = RL_Deconv(psfA_root="../PSFs/PSF_IsoLattice_560_viewA.tif",psfB_root="../PSFs/PSF_IsoLattice_560_viewB.tif",X_dim=1122,Y_dim=1122,Z_dim=108)
    rl_deconv.set_device(device)

    view1 = view1.transpose(2,1,0)/view1.max()
    view2 = view2.transpose(2, 1, 0)/view2.max()
    view1 = torch.from_numpy(view1.astype(np.float32)).to(device)
    view2 = torch.from_numpy(view2.astype(np.float32)).to(device)
    start = time.time()
    fusion = (view1+view2)/2
    for i in tqdm.tqdm(range(iter)):
        curA,curB = rl_deconv(view1,view2,fusion)
        fusion = torch.clamp(curA, min = 10e-6)*torch.clamp(curB, min = 10e-6)*fusion
        fusion = fusion/fusion.max()
    print(time.time()-start)
    print(fusion.shape)
    fusion = fusion.reshape(108,1122,1122).cpu()
    fusion = np.array(fusion*(2**16-1)).astype(np.uint16)
    tifffile.imwrite("./fusion.tif",fusion)







