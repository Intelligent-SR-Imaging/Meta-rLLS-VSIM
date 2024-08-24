"""
TV Loss in training
Created on Tue Sep 30 19:31:36 2023
@ Last Updated by Lin Yuhuan
"""
import torch
import torch.nn as nn

def tv_loss(x):
    """TV loss function"""
    z_x = x.size()[2]
    h_x = x.size()[3]
    w_x = x.size()[4]
    count_h = h_x-1
    count_w = w_x-1
    count_z = z_x - 1
    h_tv = torch.pow((x[:, :, :,1:, :] - x[:, :, :,:h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :,: , 1:] - x[:, :, :,:, :w_x - 1]), 2).sum()
    z_tv = torch.pow((x[:, :, 1:, :, :] - x[:, :, :z_x-1, :, :]), 2).sum()
    return 3*(h_tv/count_h+w_tv/count_w+5*z_tv/count_z)


class TV_Loss(nn.Module):

    def __init__(self,TVLoss_weight=0.0001):
        """TV_loss
               Args:
                  TVLoss_weight(float) : the weight of TV loss
        """
        super(TV_Loss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size=x.shape[0]
        return self.TVLoss_weight*tv_loss(x)/batch_size

class Hessian_Loss(nn.Module):
    def __init__(self,HLoss_weight=1):
        """Hessian_Loss
               Args:
                  Hessian_Loss(float) : the weight of TV loss
        """
        super(Hessian_Loss, self).__init__()
        self.HLoss_weight = HLoss_weight

    def forward(self,img):
        hess_loss = 0
        x = img[:,:,1:,:,:] - img[:,:,:-1,:,:]
        y = img[:,:, :,1:, :] - img[:, :,:,:-1,  :]
        z = img[:,:,:, :, 1:] - img[:,:,:, :, :-1]
        for tv in [x, y, z]:
            hess = tv[:,:, 1:, :, :] - tv[:,:, :-1, :, :]
            hess_loss = hess_loss + torch.mean(torch.square(hess))
            hess = tv[:,:, :, 1:, :] - tv[:,:, :, :-1, :]
            hess_loss = hess_loss + torch.mean(torch.square(hess))
            hess = tv[:,:, :, :, 1:] - tv[:, :,:, :, :-1]
            hess_loss = hess_loss + torch.mean(torch.square(hess))

        return hess_loss


