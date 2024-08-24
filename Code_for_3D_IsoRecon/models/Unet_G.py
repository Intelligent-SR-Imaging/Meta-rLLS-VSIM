"""
The Feature Extraction U part in RL-DFN
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


class double_conv(nn.Module):
    """two layers of conv layer"""
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

class last_conv(nn.Module):
    """two last conv layer"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm_layer=None, dimension = 3):
        super(last_conv, self).__init__()

        _conv = conv(dimension)

        self.convolution = nn.Sequential(
            _conv(in_channels, out_channels, kernel_size,
                stride, padding),
            norm_layer(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.convolution(x)
        return x

class triple_conv(nn.Module):
    """three layers of conv layer"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm_layer=None, dimension = 3):
        super(triple_conv, self).__init__()

        _conv = conv(dimension)

        self.convolution = nn.Sequential(
            _conv(in_channels, out_channels, kernel_size,
                stride, padding),
            norm_layer(out_channels),
            nn.ReLU(),
            _conv(out_channels, out_channels, kernel_size,
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

class Unet_deconv(nn.Module):
    """Feature Extraction U in RL-DFN
       Args::
        input_nc(int)，output_nc(int): intput/output channel size
        norm_layer : normalization layer
        dimension : input dimensions 2 or 3
        feature_input: if input feature map or not

    """
    def __init__(self, input_nc, output_nc, norm_layer=None, dimension = 3,feature_input = False):

        _maxpool = maxpool(dimension)
        _conv = conv(dimension)
        _convtranspose = convtranspose(dimension)

        super(Unet_deconv, self).__init__()
        start_nc = input_nc * 64
        if feature_input:
            start_nc = input_nc

        # Downsampling
        self.double_conv1 = double_conv(input_nc,  start_nc, 3, 1, 1, norm_layer, dimension)
        self.maxpool1 = _maxpool(2)

        self.double_conv2 = double_conv(start_nc, start_nc*2, 3, 1, 1, norm_layer, dimension)
        self.maxpool2 = _maxpool(2)

        # bottom floor
        self.bottom_layer = triple_conv(start_nc *2, start_nc*4, 3, 1, 1, norm_layer, dimension)

        # Upsampling = transposed convolution
        self.t_conv2 = _convtranspose (start_nc*4, start_nc*2, 2, 2)
        self.ex_double_conv2 = double_conv(start_nc*4, start_nc*2, 3, 1, 1, norm_layer, dimension)

        self.t_conv1 = _convtranspose(start_nc*2, start_nc, 2, 2)
        self.ex_conv1_1 = last_conv(start_nc*2, start_nc, 3, 1, 1, norm_layer, dimension)

        # last stage
        self.one_by_one = _conv(start_nc, output_nc, 1, 1, 0)
        self.one_by_one_2 = _conv(output_nc, output_nc, 1, 1, 0)

        self.leakyRelu = nn.Softplus()


    def forward(self, inputs):

        # Contracting Path
        conv1 = self.double_conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.double_conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        # bottom floor
        conv_bottom = self.bottom_layer(maxpool2)

        t_conv2 = self.t_conv2(conv_bottom)

        cat2 = torch.cat([conv2, t_conv2], 1)

        ex_conv2 = self.ex_double_conv2(cat2)

        t_conv1 = self.t_conv1(ex_conv2)
        cat1 = torch.cat([conv1, t_conv1], 1)
        ex_conv1 = self.ex_conv1_1(cat1)

        one_by_one = self.one_by_one(ex_conv1)
        one_by_one_2 = self.one_by_one_2(one_by_one)
        last_val = self.leakyRelu(one_by_one_2)


        return last_val