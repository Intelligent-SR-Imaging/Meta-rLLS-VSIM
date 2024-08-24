"""
The Feature Extraction A/B part in RL-DFN
Created on Tue Sep 30 19:31:36 2023
@ Last Updated by Lin Yuhuan
"""
import torch.nn as nn

def convtranspose(dimension):
    if dimension == 2:
        return nn.ConvTranspose2d

    elif dimension == 3:
        return nn.ConvTranspose3d

    else:
        raise Exception('Invalid image dimension.')


def conv(dimension):
    if dimension == 2:
        return nn.Conv2d

    elif dimension == 3:
        return nn.Conv3d

    else:
        raise Exception('Invalid image dimension.')

def maxpool(dimension):

    if dimension == 2:
        return nn.MaxPool2d

    elif dimension == 3:
        return nn.MaxPool3d

    else:
        raise Exception('Invalid image dimension.')


class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm_layer=None, dimension = 3):
        super(double_conv, self).__init__()
        _conv = conv(dimension)

        self.convolution = nn.Sequential(
            _conv(in_channels, out_channels, kernel_size,
                stride, padding),
            norm_layer(out_channels),
            nn.ReLU(),
            _conv(out_channels, out_channels, kernel_size,
                stride, padding),
            norm_layer(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.convolution(x)
        return x

class Feature_Net(nn.Module):
    """Feature Extraction A/B in RL-DFN
       Args::
        input_nc(int)，output_nc(int): intput/output channel size
        norm_layer : normalization layer
        dimension : input dimensions 2 or 3
    """

    def __init__(self, input_nc, output_nc, norm_layer=None, dimension = 3):


        _conv = conv(dimension)


        super(Feature_Net, self).__init__()
        start_nc = input_nc * 64

        self.first_conv = _conv(input_nc,start_nc , 1, 1, 0)

        self.mid_conv = double_conv(start_nc, start_nc, 3, 1, 1, norm_layer, dimension)

        self.last_conv = _conv(start_nc, output_nc, 1, 1, 0)

        self.leakyRelu = nn.Sigmoid()


    def forward(self, inputs):

        # Contracting Path

        x = self.first_conv(inputs)
        x = self.mid_conv(x)
        x = self.last_conv(x)
        last_val = self.leakyRelu(x)


        return last_val