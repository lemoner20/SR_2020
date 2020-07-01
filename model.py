#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Transform_Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Transform_Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class Transform_Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Transform_Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


class TVLoss(nn.Module):

    def __init__(self, tv_weight):
        super(TVLoss, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, x):

        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]),2).sum()
        return self.tv_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class ConvBlock(nn.Module):

    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(64, affine=True)
        self.instance_norm2 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.instance_norm1(self.conv1(x)))
        y = self.instance_norm2(self.conv2(y)) + x
        return y


class Enhance_Generator(nn.Module):

    def __init__(self):
        super(Enhance_Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.blocks = nn.Sequential(
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
        )
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.instance_norm = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, 1, padding=0)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        temp = x
        x = self.blocks(x)
        x = self.instance_norm(self.conv2(x)) + temp
        x = F.relu(self.conv3(x))
        x = F.tanh(self.conv4(x))

        return x


class Enhance_DiscriminatorC(nn.Module):

    def __init__(self, input_ch):
        super(Enhance_DiscriminatorC, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_ch, 48, 11, stride=4, padding=5),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(48, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(128, affine=True),
            nn.Conv2d(128, 192, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(192, affine=True),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(192, affine=True),
            nn.Conv2d(192, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(128, affine=True),
        )

        self.fc = nn.Linear(128*9*9, 1024)
        self.out = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.conv_layers(x)
        #print(x.shape)
        x = x.view(-1, 128*9*9)
        #print(x.shape)
        x = F.leaky_relu(self.fc(x), negative_slope=0.2)
        x = F.softmax(self.out(x))
        return x


class Enhance_DiscriminatorT(nn.Module):

    def __init__(self, input_ch):
        super(Enhance_DiscriminatorT, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_ch, 48, 11, stride=4, padding=5),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(48, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(128, affine=True),
            nn.Conv2d(128, 192, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(192, affine=True),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(192, affine=True),
            nn.Conv2d(192, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(128, affine=True),
        )

        self.fc = nn.Linear(128*8*8, 1024)
        self.out = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.conv_layers(x)
        #print(x.shape)
        x = x.view(-1, 128*8*8)
        #print(x.shape)
        x = F.leaky_relu(self.fc(x), negative_slope=0.2)
        x = F.softmax(self.out(x))
        return x


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = [[0.03797616, 0.044863533, 0.03797616],
                  [0.044863533, 0.053, 0.044863533],
                  [0.03797616, 0.044863533, 0.03797616]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.weight = kernel.cuda()

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=2)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=2)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class GrayLayer(nn.Module):

    def __init__(self):
        super(GrayLayer, self).__init__()

    def forward(self, x):
        result = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        # print(result.unsqueeze(1).shape)
        return result.unsqueeze(1)


class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        self.model = vgg19_bn(True).features.cuda()
        self.mean = torch.Tensor([123.68,  116.779,  103.939]).cuda().view(1,3,1,1)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x*255 - self.mean
        #print(type(x), x.shape)
        x = self.model(x)
        return x
