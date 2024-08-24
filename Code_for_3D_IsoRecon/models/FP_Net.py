"""
The RNL block implementation
Created on Tue Sep 30 19:31:36 2023
@ Last Updated by Lin Yuhuan
"""
import torch
import torch.nn as nn

def convtranspose(dimension):
    """convtranspose layer"""
    if dimension == 2:
        return nn.ConvTranspose2d

    elif dimension == 3:
        return nn.ConvTranspose3d

    else:
        raise Exception('Invalid image dimension.')


def conv(dimension):
    """conv layer"""
    if dimension == 2:
        return nn.Conv2d

    elif dimension == 3:
        return nn.Conv3d

    else:
        raise Exception('Invalid image dimension.')

def maxpool(dimension):
    """maxpool layer"""
    if dimension == 2:
        return nn.MaxPool2d

    elif dimension == 3:
        return nn.MaxPool3d

    else:
        raise Exception('Invalid image dimension.')


class conv_layer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm_layer=None, dimension = 3):
        super(conv_layer, self).__init__()
        _conv = conv(dimension)

        self.convolution = nn.Sequential(
            _conv(in_channels, out_channels, kernel_size,
                stride, padding),
            norm_layer(out_channels),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.convolution(x)
        return x

class FP1_Net(nn.Module):

    def __init__(self, norm_layer=None, dimension = 3):


        _conv = conv(dimension)


        super(FP1_Net, self).__init__()


        self.first_conv1 = conv_layer(1, 16 ,kernel_size= 3,stride= 1,padding= 1,norm_layer=norm_layer)

        self.first_conv2 = conv_layer(1, 16 ,kernel_size= 3,stride= 1,padding= 1,norm_layer=norm_layer)

        self.last_conv = conv_layer(32, 1 ,kernel_size= 3,stride= 1,padding= 1,norm_layer=norm_layer)




    def forward(self, inputs):

        # Contracting Path

        x1 = self.first_conv1(inputs)
        x2 = self.first_conv2(inputs)
        lastval = self.last_conv(torch.cat((x1,x2),dim=1))
        last_val = (lastval+inputs)/2


        return last_val


class BP1_Net(nn.Module):

    def __init__(self, norm_layer=None, dimension=3):
        _conv = conv(dimension)

        super(BP1_Net, self).__init__()

        self.first_conv1 = conv_layer(1, 16, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer)

        self.first_conv2 = conv_layer(1, 16, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer)

        self.last_conv = conv_layer(32, 1, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer)

    def forward(self, inputs):
        # Contracting Path

        x1 = self.first_conv1(inputs)
        x2 = self.first_conv2(inputs)
        lastval = self.last_conv(torch.cat((x1, x2), dim=1))


        return lastval

class RNL(nn.Module):
    def __init__(self, norm_layer=None,alpha = 0.0001):


        super(RNL, self).__init__()

        self.FP1 = FP1_Net(norm_layer=norm_layer,dimension=3)
        self.BP1 = BP1_Net(norm_layer=norm_layer, dimension=3)
        self.FP2 = FP1_Net(norm_layer=norm_layer, dimension=3)
        self.BP2 = BP1_Net(norm_layer=norm_layer, dimension=3)
        self.alpha = alpha

    def forward(self,viewA,viewB,fusion):
        fp1 = self.FP1(fusion)
        DV1 = viewA/(fp1+self.alpha)
        curA = self.BP1(DV1)

        fp2 = self.FP2(fusion)
        DV2 = viewB / (fp2 + self.alpha)
        curB = self.BP2(DV2)

        return curA,curB
