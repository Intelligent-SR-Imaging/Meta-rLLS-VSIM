"""
Patch-GAN-D in RL-DFN
Created on Tue Sep 30 19:31:36 2023
@ Last Updated by Lin Yuhuan
"""
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


######################END OF PETER'S IMPLEMENTATION##############################################
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=None, use_sigmoid = True, dimension =3, use_bias = True):
        """Construct a PatchGAN discriminator
           Args::
            input_nc (int): the number of channels in input images
            ndf (int): the number of filters in the last conv layer
            n_layers (int): the number of conv layers in the discriminator
            norm_layer : normalization layer
            use_sigmoid : activation function
            dimension : input dimensions 2 or 3
            use_bias : use bias in conv layer or not

        """

        _conv = conv(dimension)

        super(NLayerDiscriminator, self).__init__()


        kw = 4
        padw = 1
        sequence = [_conv(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1


        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            sequence += [
                _conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        sequence += [
            _conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [_conv(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map

        if use_sigmoid:
            print ("Using sigmoid in the last layer of Discriminator. Note that LSGAN may work well with this loss.")
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        # is_cuda = next(self.model.parameters()).is_cuda
        return self.model(input)