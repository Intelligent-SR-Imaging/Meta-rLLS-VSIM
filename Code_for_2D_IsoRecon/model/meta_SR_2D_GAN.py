"""
the model of meta-VSI-SR
Created on Tue Sep 30 19:31:36 2023
@ Last Updated by Lin Yuhuan
"""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math


# Channel Attention layer (CA)
class CA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA, self).__init__()
        self.channel = channel
        self.vars = nn.ParameterList()

        param = [channel // reduction, channel, 1, 1]
        w = nn.Parameter(torch.ones(*param))
        # torch.nn.init.kaiming_normal_(w)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.vars.append(w)
        bias = nn.Parameter(torch.zeros(param[0]))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        self.vars.append(bias)

        param = [channel, channel // reduction, 1, 1]
        w = nn.Parameter(torch.ones(*param))
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.vars.append(w)
        bias = nn.Parameter(torch.zeros(param[0]))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        self.vars.append(bias)

    def extra_repr(self, reduction=16):
        info = ''
        tmp = 'AdaptiveAvg_Pool2d: (output_size: 1)'
        info += tmp + '\n'
        tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:1, stride:1, padding:same)' % (self.channel, self.channel // reduction)
        info += tmp + '\n'
        tmp = 'ReLU: '
        info += tmp + '\n'
        tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:1, stride:1, padding:same)' % (self.channel // reduction, self.channel)
        info += tmp + '\n'
        tmp = 'Sigmoid: '
        info += tmp + '\n'
        return info

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        y = F.adaptive_avg_pool2d(x, 1)
        w, b = vars[0], vars[1]
        y = F.conv2d(y, w, b, padding=0)
        y = F.relu(y, inplace=True)
        w, b = vars[2], vars[3]
        y = F.conv2d(y, w, b, padding=0)
        y = torch.sigmoid(y)
        return x*y

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars


class RCAB(nn.Module):
    def __init__(self, channel):
        super(RCAB, self).__init__()
        self.channel = channel
        self.vars = nn.ParameterList()

        param = [channel, channel, 3, 3]
        w = nn.Parameter(torch.ones(*param))
        # nn.init.kaiming_normal_(w)
        # self.vars.append(w)
        # self.vars.append(nn.Parameter(torch.zeros(param[0])))
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.vars.append(w)
        bias = nn.Parameter(torch.zeros(param[0]))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        self.vars.append(bias)

        param = [channel, channel, 3, 3]
        w = nn.Parameter(torch.ones(*param))
        # nn.init.kaiming_normal_(w)
        # self.vars.append(w)
        # self.vars.append(nn.Parameter(torch.zeros(param[0])))
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.vars.append(w)
        bias = nn.Parameter(torch.zeros(param[0]))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        self.vars.append(bias)

        self.ca = CA(channel)
        for i in range(len(self.ca.vars)):
            self.vars.append(self.ca.vars[i])

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        w, b = vars[0], vars[1]
        y = F.conv2d(x, w, b, padding=1)
        y = F.leaky_relu(y, 0.2, inplace=True)
        w, b = vars[2], vars[3]
        y = F.conv2d(y, w, b, padding=1)
        y = F.leaky_relu(y, 0.2, inplace=True)
        y = self.ca(y, vars[4:])
        return x+y

    def extra_repr(self):
        info = ''
        tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:3, stride:1, padding:same)' % (self.channel, self.channel)
        info += tmp + '\n'
        tmp = 'LeakyReLU: negative_slope=0.2, inplace=True'
        info += tmp + '\n'
        tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:3, stride:1, padding:same)' % (self.channel, self.channel)
        info += tmp + '\n'
        tmp = 'LeakyReLU: negative_slope=0.2, inplace=True'
        info += tmp + '\n'
        info += self.ca.extra_repr()
        return info

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars


class ResidualGroup(nn.Module):
    def __init__(self, channel, n_RCAB=4):
        super(ResidualGroup, self).__init__()
        self.n_RCAB = n_RCAB
        self.channel = channel
        self.vars = nn.ParameterList()
        self.rcab = []

        for j in range(n_RCAB):
            self.rcab.append(RCAB(channel))
            for i in range(len(self.rcab[j].vars)):
                self.vars.append(self.rcab[j].vars[i])

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars

        for j in range(self.n_RCAB):
            k = len(self.rcab[j].vars)
            x = self.rcab[j](x, vars[j*k:(j+1)*k])
        return x

    def extra_repr(self):
        info = ''
        for j in range(self.n_RCAB):
            info += self.rcab[j].extra_repr()
        return info

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars


class Generator(nn.Module):
    def __init__(self, in_channel=7, n_ResGroup=4, n_RCAB=4, out_channel=1):
        super(Generator, self).__init__()
        self.n_ResGroup = n_ResGroup
        self.vars = nn.ParameterList()
        self.resgroup = []

        param = [64, in_channel, 3, 3]
        w = nn.Parameter(torch.ones(*param))
        # nn.init.kaiming_normal_(w)
        # self.vars.append(w)
        # self.vars.append(nn.Parameter(torch.zeros(param[0])))
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.vars.append(w)
        bias = nn.Parameter(torch.zeros(param[0]))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        self.vars.append(bias)

        for j in range(n_ResGroup):
            self.resgroup.append(ResidualGroup(64, n_RCAB=n_RCAB))
            for i in range(len(self.resgroup[j].vars)):
                self.vars.append(self.resgroup[j].vars[i])

        param = [256, 64, 3, 3]
        w = nn.Parameter(torch.ones(*param))
        # nn.init.kaiming_normal_(w)
        # self.vars.append(w)
        # self.vars.append(nn.Parameter(torch.zeros(param[0])))
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.vars.append(w)
        bias = nn.Parameter(torch.zeros(param[0]))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        self.vars.append(bias)

        param = [out_channel, 256, 3, 3]
        w = nn.Parameter(torch.ones(*param))
        # nn.init.kaiming_normal_(w)
        # self.vars.append(w)
        # self.vars.append(nn.Parameter(torch.zeros(param[0])))
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.vars.append(w)
        bias = nn.Parameter(torch.zeros(param[0]))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        self.vars.append(bias)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars

        w, b = vars[0], vars[1]
        x = F.conv2d(x, w, b, padding=1)
        for j in range(self.n_ResGroup):
            k = len(self.resgroup[j].vars)
            x = self.resgroup[j](x, vars[2+j*k:2+(j+1)*k])
        # x = F.upsample(x, scale_factor=(1, 1.5, 1.5), mode='trilinear')
        x = F.interpolate(x, scale_factor=(1.5, 1.5), mode='bilinear', align_corners=False, recompute_scale_factor=False)
        k = 2+self.n_ResGroup*len(self.resgroup[0].vars)
        w, b = vars[k], vars[k+1]
        x = F.conv2d(x, w, b, padding=1)
        x = F.leaky_relu(x, 0.2, inplace=True)
        w, b = vars[k+2], vars[k+3]
        x = F.conv2d(x, w, b, padding=1)
        x = F.leaky_relu(x, 0.2, inplace=True)
        return x

    def extra_repr(self):
        info = ''
        tmp = 'conv2d:(ch_in:7, ch_out:64, k:3, stride:1, padding:same)'
        info += tmp + '\n'
        for j in range(self.n_ResGroup):
            info += self.resgroup[j].extra_repr()
        tmp = 'Upsampling 1.5x: bilinear'
        info += tmp + '\n'
        tmp = 'conv2d:(ch_in:64, ch_out:256, k:3, stride:1, padding:same)'
        info += tmp + '\n'
        tmp = 'LeakyReLU: negative_slope=0.2, inplace=True'
        info += tmp + '\n'
        tmp = 'conv2d:(ch_in:256, ch_out:1, k:3, stride:1, padding:same)'
        info += tmp + '\n'
        tmp = 'LeakyReLU: negative_slope=0.2, inplace=True'
        info += tmp + '\n'
        return info

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars


class conv_block2d(nn.Module):
    def __init__(self, channel_input, channel_output):
        super(conv_block2d, self).__init__()
        self.channel_in = channel_input
        self.channel_out = channel_output
        self.vars = nn.ParameterList()
        param = [channel_output[0], channel_input, 3, 3]
        w = nn.Parameter(torch.ones(*param))
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.vars.append(w)
        bias = nn.Parameter(torch.zeros(param[0]))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        self.vars.append(bias)

        param = [channel_output[1], channel_output[0], 3, 3]
        w = nn.Parameter(torch.ones(*param))
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.vars.append(w)
        bias = nn.Parameter(torch.zeros(param[0]))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        self.vars.append(bias)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        w, b = vars[0], vars[1]
        x = F.conv2d(x, w, b, padding='same')
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)
        w, b = vars[2], vars[3]
        x = F.conv2d(x, w, b, padding='same')
        x = F.leaky_relu(x, 0.1, inplace=True)
        return x

    def extra_repr(self):
        info = ''
        tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:3, stride:1, padding:same)' % (self.channel_in, self.channel_out[0])
        info += tmp + '\n'
        tmp = 'LeakyReLU: negative_slope=0.1, inplace=True'
        info += tmp + '\n'
        tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:3, stride:1, padding:same)' % (self.channel_out[0], self.channel_out[1])
        info += tmp + '\n'
        tmp = 'LeakyReLU: negative_slope=0.1, inplace=True'
        info += tmp + '\n'
        return info

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars


class Discriminator(nn.Module):
    def __init__(self, in_ch=1):
        super(Discriminator, self).__init__()
        self.vars = nn.ParameterList()
        param = [32, in_ch, 3, 3]
        w = nn.Parameter(torch.ones(*param))
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.vars.append(w)
        bias = nn.Parameter(torch.zeros(param[0]))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        self.vars.append(bias)

        self.conv2d1 = conv_block2d(32, (32, 32))
        for i in range(len(self.conv2d1.vars)):
            self.vars.append(self.conv2d1.vars[i])
        self.conv2d2 = conv_block2d(32, (64, 64))
        for i in range(len(self.conv2d2.vars)):
            self.vars.append(self.conv2d2.vars[i])
        self.conv2d3 = conv_block2d(64, (128, 128))
        for i in range(len(self.conv2d3.vars)):
            self.vars.append(self.conv2d3.vars[i])
        self.conv2d4 = conv_block2d(128, (256, 256))
        for i in range(len(self.conv2d4.vars)):
            self.vars.append(self.conv2d4.vars[i])
        self.conv2d5 = conv_block2d(256, (512, 512))
        for i in range(len(self.conv2d5.vars)):
            self.vars.append(self.conv2d5.vars[i])

        param = [256, 512]
        w = nn.Parameter(torch.ones(*param))
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.vars.append(w)
        bias = nn.Parameter(torch.zeros(param[0]))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        self.vars.append(bias)

        param = [1, 256]
        w = nn.Parameter(torch.ones(*param))
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.vars.append(w)
        bias = nn.Parameter(torch.zeros(param[0]))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        self.vars.append(bias)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        w, b = vars[0], vars[1]
        x = F.conv2d(x, w, b, padding='same')
        x = F.leaky_relu(x, 0.1, inplace=True)

        k = len(self.conv2d1.vars)
        x1 = self.conv2d1(x, vars[2: 2 + k])
        x2 = self.conv2d2(x1, vars[2 + k: 2 + 2 * k])
        x3 = self.conv2d3(x2, vars[2 + 2 * k: 2 + 3 * k])
        x4 = self.conv2d4(x3, vars[2 + 3 * k: 2 + 4 * k])
        x5 = self.conv2d5(x4, vars[2 + 4 * k: 2 + 5 * k])

        x6 = F.adaptive_avg_pool2d(x5, 1)
        y0 = x6.view(x6.size(0), -1)

        w, b = vars[2 + 5 * k], vars[3 + 5 * k]
        y1 = F.linear(y0, w, b)
        y1 = F.leaky_relu(y1, 0.1, inplace=True)
        w, b = vars[4 + 5 * k], vars[5 + 5 * k]
        outputs = F.linear(y1, w, b)
        outputs = torch.sigmoid(outputs)

        return outputs

    def extra_repr(self):
        info = ''
        tmp = 'conv2d:(ch_in:1, ch_out:32, k:3, stride:1, padding:same)'
        info += tmp + '\n'
        tmp = 'LeakyReLU: negative_slope=0.1, inplace=True'
        info += tmp + '\n'

        info += self.conv2d1.extra_repr()
        info += self.conv2d2.extra_repr()
        info += self.conv2d3.extra_repr()
        info += self.conv2d4.extra_repr()
        info += self.conv2d5.extra_repr()

        tmp = 'global_avg_pool2d: (channel:512, 1, 1)'
        info += tmp + '\n'
        tmp = 'Flatten: (channel:512)'
        info += tmp + '\n'

        tmp = 'linear: (in:512, out:256)'
        info += tmp + '\n'
        tmp = 'LeakyReLU: negative_slope=0.1, inplace=True'
        info += tmp + '\n'

        tmp = 'linear: (in:256, out:1)'
        info += tmp + '\n'
        tmp = 'Sigmoid'
        info += tmp + '\n'
        return info

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars




if __name__ == '__main__':
    input1 = torch.randn(1, 7, 64, 64)
    g = Generator(in_channel=7, out_channel=3)
    num_params = 0
    for param in g.parameters():
        num_params += param.nelement()
    print('numbers of g_parameter is {}'.format(num_params))
    out = g(input1)
    print('output shape is {}'.format(out.shape))
    # print(g.extra_repr())

    d = Discriminator(in_ch=3)
    num_params = 0
    for param in d.parameters():
        num_params += param.nelement()
    print('numbers of d_parameter is {}'.format(num_params))
    out1 = d(out)
    print('d_output shape is {}'.format(out1.shape))
    # print(d.extra_repr())






