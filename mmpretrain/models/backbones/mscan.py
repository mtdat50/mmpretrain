# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmengine.registry import HOOKS
from mmengine.hooks import Hook

from mmpretrain.registry import MODELS
from mmseg.models.decode_heads.ham_head import Hamburger, CustomHamburger


class Mlp(BaseModule):
    """Multi Layer Perceptron (MLP) Module.

    Args:
        in_features (int): The dimension of input features.
        hidden_features (int): The dimension of hidden features.
            Defaults: None.
        out_features (int): The dimension of output features.
            Defaults: None.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        drop (float): The number of dropout rate in MLP block.
            Defaults: 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.,
                 channel_attention_type=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=True,
            groups=hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        match channel_attention_type:
            case "SE":
                self.channel_attention = SqueezeExcitation(hidden_features, hidden_features // 32)
            case "ECA":
                    t = (math.log2(hidden_features) + 1) // 2
                    k = t if t % 2 else t + 1
                    self.channel_attention = eca_layer(k_size=int(k))
            case "CBAM":
                self.channel_attention = CAM(hidden_features, r=1)
            case _:
                self.channel_attention = nn.Identity()

    def forward(self, x):
        """Forward function."""

        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = x + self.channel_attention(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(BaseModule):
    """Stem Block at the beginning of Semantic Branch.

    Args:
        in_channels (int): The dimension of input channels.
        out_channels (int): The dimension of output channels.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            build_activation_layer(act_cfg),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        """Forward function."""

        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class MSCAAttention(BaseModule):
    """Attention Module in Multi-Scale Convolutional Attention Module (MSCA).

    Args:
        channels (int): The dimension of channels.
        kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
    """

    def __init__(self,
                 channels,
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 paddings=[2, [0, 3], [0, 5], [0, 10]]):
        super().__init__()
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        for i, (kernel_size,
                padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels))
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        """Forward function."""

        u = x.clone()

        attn = self.conv0(x)

        # Multi-Scale Feature extraction
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        # Channel Mixing
        attn = self.conv3(attn)

        # Convolutional Attention
        x = attn * u

        return x


class CustomMSCAAttention(BaseModule):
    """Attention Module in Multi-Scale Convolutional Attention Module (MSCA).

    Args:
        channels (int): The dimension of channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels) #3
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels) #5, 7
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #7, 13
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #7, 19
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #7, 25
        self.channel_mixing = nn.Conv2d(channels, channels, 1)


    def forward(self, x):
        u = x.clone()

        # Multi-Scale Feature extraction
        attn1 = self.conv1(self.conv0(x))
        attn2 = self.conv2(attn1)
        attn3 = self.conv3(attn2)
        attn4 = self.conv4(attn3)

        attn = attn1 + attn2 + attn3 + attn4
        attn = self.channel_mixing(attn)

        # Convolutional Attention
        out = attn * u

        return out


class CustomMSCAAttention2(CustomMSCAAttention):
    """ With skip connections between the multi-scale feature extraction layers

    Args:
        channels (int): The dimension of channels.
    """

    def forward(self, x):
        u = x.clone()

        # Multi-Scale Feature extraction
        attn0 = self.conv0(x)
        attn1 = self.conv1(attn0)
        sum1 = attn1 + attn0
        attn2 = self.conv2(sum1)
        sum2 = attn2 + sum1
        attn3 = self.conv3(sum2)
        sum3 = attn3 + sum2
        attn4 = self.conv4(sum3)

        attn = attn1 + attn2 + attn3 + attn4
        attn = self.channel_mixing(attn)

        # Convolutional Attention
        out = attn * u

        return out


class CustomMSCAAttention3(BaseModule):
    """Max dilation

    Args:
        channels (int): The dimension of channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels) #3
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #9
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=9, dilation=9, groups=channels) #27
        self.channel_mixing = nn.Conv2d(channels, channels, 1)


    def forward(self, x):
        u = x.clone()

        # Multi-Scale Feature extraction
        attn0 = self.conv0(x)
        attn1 = self.conv1(attn0)
        sum1 = attn1 + attn0
        attn2 = self.conv2(sum1)

        attn = attn0 + attn1 + attn2
        attn = self.channel_mixing(attn)

        # Convolutional Attention
        out = attn * u

        return out


class CustomMSCAAttention4(BaseModule):
    """ With skip connections between the multi-scale feature extraction layers

    Args:
        channels (int): The dimension of channels.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels) #3
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #9
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #15
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #21
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #27
        self.channel_mixing = nn.Conv2d(4 * channels, channels, 1)

    def forward(self, x):
        u = x.clone()

        # Multi-Scale Feature extraction
        attn0 = self.conv0(x)
        attn1 = self.conv1(attn0)
        sum1 = attn1 + attn0
        attn2 = self.conv2(sum1)
        sum2 = attn2 + sum1
        attn3 = self.conv3(sum2)
        sum3 = attn3 + sum2
        attn4 = self.conv4(sum3)

        attn = torch.cat([attn1, attn2, attn3, attn4], dim=1)
        attn = self.channel_mixing(attn)

        # Convolutional Attention
        out = attn * u

        return out


class CustomMSCAAttention5(CustomMSCAAttention2):
    """ With skip connections between the multi-scale feature extraction layers

    Args:
        channels (int): The dimension of channels.
    """
    def __init__(self, channels):
        super().__init__(channels)
        self.weight = nn.Parameter(torch.ones(4))

    def forward(self, x):
        u = x.clone()

        # Multi-Scale Feature extraction
        attn0 = self.conv0(x)
        attn1 = self.conv1(attn0)
        sum1 = attn1 + attn0
        attn2 = self.conv2(sum1)
        sum2 = attn2 + sum1
        attn3 = self.conv3(sum2)
        sum3 = attn3 + sum2
        attn4 = self.conv4(sum3)

        attn = (
            attn1 * self.weight[0] +
            attn2 * self.weight[1] +
            attn3 * self.weight[2] +
            attn4 * self.weight[3]
        )
        attn = self.channel_mixing(attn)

        # Convolutional Attention
        out = attn * u

        return out


class CustomMSCAAttention6(BaseModule):
    """Attention Module in Multi-Scale Convolutional Attention Module (MSCA).

    Args:
        channels (int): The dimension of channels.
        kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
    """

    def __init__(self,
                 channels,
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 paddings=[2, [0, 3], [0, 5], [0, 10]]):
        super().__init__()
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        for i, (kernel_size,
                padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels))
        self.weight = nn.Parameter(torch.ones(4))
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        """Forward function."""

        u = x.clone()

        attn = self.conv0(x)

        # Multi-Scale Feature extraction
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = (
            self.weight[0] * attn +
            self.weight[1] * attn_0 +
            self.weight[2] * attn_1 +
            self.weight[3] * attn_2
        )
        # Channel Mixing
        attn = self.conv3(attn)

        # Convolutional Attention
        x = attn * u

        return x


class CustomMSCAAttention7(CustomMSCAAttention):
    def __init__(self, channels):
        super().__init__(channels)
        self.weight = nn.Parameter(torch.ones(4))

    def forward(self, x):
        u = x.clone()

        # Multi-Scale Feature extraction
        attn1 = self.conv1(self.conv0(x))
        attn2 = self.conv2(attn1)
        attn3 = self.conv3(attn2)
        attn4 = self.conv4(attn3)

        attn = (
            self.weight[0] * attn1 +
            self.weight[1] * attn2 +
            self.weight[2] * attn3 +
            self.weight[3] * attn4
        )
        attn = self.channel_mixing(attn)

        # Convolutional Attention
        out = attn * u

        return out


class CustomMSCAAttention8(BaseModule):
    """half dilation

    Args:
        channels (int): The dimension of channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels) #3
        # self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels) #5, 7
        # self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4, groups=channels) #9, 15
        # self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=6, dilation=6, groups=channels) #13, 27
        # self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=10, dilation=10, groups=channels) #21, 47
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels) #5, 7
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #7, 13
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=7, dilation=7, groups=channels) #15, 27
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=13, dilation=13, groups=channels) #27, 53
        self.weight = nn.Parameter(torch.ones(4))
        self.channel_mixing = nn.Conv2d(channels, channels, 1)


    def forward(self, x):
        u = x.clone()

        # Multi-Scale Feature extraction
        attn0 = self.conv0(x)
        attn1 = self.conv1(attn0)
        sum1 = attn1 + attn0
        attn2 = self.conv2(sum1)
        sum2 = sum1 + attn2
        attn3 = self.conv3(sum2)
        sum3 = sum2 + attn3
        attn4 = self.conv4(sum3)

        attn = (
            self.weight[0] * attn1 +
            self.weight[1] * attn2 +
            self.weight[2] * attn3 +
            self.weight[3] * attn4
        )
        attn = self.channel_mixing(attn)

        # Convolutional Attention
        out = attn * u

        return out


class CustomMSCAAttention9(BaseModule):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels) #3
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels) #5, 7
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #7, 13
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=7, dilation=7, groups=channels) #15, 27
        self.weight = nn.Parameter(torch.ones(3))
        self.channel_mixing = nn.Conv2d(channels, channels, 1)


    def forward(self, x):
        u = x.clone()

        # Multi-Scale Feature extraction
        attn0 = self.conv0(x)
        attn1 = self.conv1(attn0)
        sum1 = attn1 + attn0
        attn2 = self.conv2(sum1)
        sum2 = sum1 + attn2
        attn3 = self.conv3(sum2)

        attn = (
            self.weight[0] * attn1 +
            self.weight[1] * attn2 +
            self.weight[2] * attn3
        )
        attn = self.channel_mixing(attn)

        # Convolutional Attention
        out = attn * u

        return out

class CustomMSCAAttention10(CustomMSCAAttention8):
    def forward(self, x):
        u = x.clone()

        # Multi-Scale Feature extraction
        attn0 = self.conv0(x)
        attn1 = self.conv1(attn0)
        attn2 = self.conv2(attn1)
        attn3 = self.conv3(attn2)
        attn4 = self.conv4(attn3)

        attn = (
            self.weight[0] * attn1 +
            self.weight[1] * attn2 +
            self.weight[2] * attn3 +
            self.weight[3] * attn4
        )
        attn = self.channel_mixing(attn)

        # Convolutional Attention
        out = attn * u

        return out


class CustomMSCAAttention11(CustomMSCAAttention8):
    def forward(self, x):
        u = x.clone()

        # Multi-Scale Feature extraction
        attn0 = self.conv0(x)
        attn1 = self.conv1(attn0)
        attn2 = self.conv2(attn1)
        attn3 = self.conv3(attn2)
        attn4 = self.conv4(attn3)

        attn = attn1 + attn2 + attn3 + attn4
        attn = self.channel_mixing(attn)

        # Convolutional Attention
        out = attn * u

        return out


class CustomMSCAAttention12(BaseModule):
    """ With skip connections between the multi-scale feature extraction layers

    Args:
        channels (int): The dimension of channels.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels) #3
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels) #5, 7
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #7, 13
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4, groups=channels) #9, 21
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=5, dilation=5, groups=channels) #11, 31
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, padding=6, dilation=6, groups=channels) #13, 43
        self.weight = nn.Parameter(torch.ones(5))
        self.channel_mixing = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        u = x.clone()

        # Multi-Scale Feature extraction
        attn0 = self.conv0(x)
        attn1 = self.conv1(attn0)
        sum1 = attn1 + attn0
        attn2 = self.conv2(sum1)
        sum2 = attn2 + sum1
        attn3 = self.conv3(sum2)
        sum3 = attn3 + sum2
        attn4 = self.conv4(sum3)
        sum4 = attn4 + sum3
        attn5 = self.conv5(sum4)

        attn = (
            attn1 * self.weight[0] +
            attn2 * self.weight[1] +
            attn3 * self.weight[2] +
            attn4 * self.weight[3] +
            attn5 * self.weight[4]
        )
        attn = self.channel_mixing(attn)

        # Convolutional Attention
        out = attn * u

        return out


class CustomMSCAAttention13(CustomMSCAAttention8):
    """ With skip connections between the multi-scale feature extraction layers

    Args:
        channels (int): The dimension of channels.
    """
    def __init__(self, channels):
        super().__init__(channels)
        # self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #7, 13
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #7, 19
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #7, 25
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #7, 31
        self.conv6 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #7, 37
        self.conv7 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels) #7, 43
        self.weight = nn.Parameter(torch.ones(7))
        self.channel_mixing = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        u = x.clone()

        # Multi-Scale Feature extraction
        attn0 = self.conv0(x)
        attn1 = self.conv1(attn0)
        sum1 = attn1 + attn0
        attn2 = self.conv2(sum1)
        sum2 = attn2 + sum1
        attn3 = self.conv3(sum2)
        sum3 = attn3 + sum2
        attn4 = self.conv4(sum3)
        sum4 = attn4 + sum3
        attn5 = self.conv5(sum4)
        sum5 = attn5 + sum4
        attn6 = self.conv6(sum5)
        sum6 = attn6 + sum5
        attn7 = self.conv7(sum6)

        attn = (
            attn1 * self.weight[0] +
            attn2 * self.weight[1] +
            attn3 * self.weight[2] +
            attn4 * self.weight[3] +
            attn5 * self.weight[4] +
            attn6 * self.weight[5] +
            attn7 * self.weight[6]
        )
        attn = self.channel_mixing(attn)

        # Convolutional Attention
        out = attn * u

        return out


class CustomMSCAAttention15(MSCAAttention):
    def __init__(self, channels):
        super().__init__(channels)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Unflatten(1, (1, channels))
        )
        self.weighting = nn.Sequential(
            # nn.Conv1d(1, 1, channels // 4, padding=channels // 8 - 1, stride=channels // 4),
            # nn.ReLU(inplace=True),
            # nn.Linear(16 + (channels % 4 != 0), 4),
            # nn.Softmax(dim=-1)
            nn.Linear(channels * 4, channels // 4),
            nn.ReLU(inplace=True),
            # nn.Linear(16 + (channels % 4 != 0), 4),
            nn.Linear(channels // 4, 4),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """Forward function."""

        u = x.clone()

        attn = self.conv0(x)

        # Multi-Scale Feature extraction
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        # attn = attn + attn_0 + attn_1 + attn_2
        pooled = torch.cat([self.gap(attn), self.gap(attn_0), self.gap(attn_1), self.gap(attn_2)], dim=2)
        weights = self.weighting(pooled)
        attn = (
            weights[:, 0, 0].view(-1, 1, 1, 1) * attn +
            weights[:, 0, 1].view(-1, 1, 1, 1) * attn_0 +
            weights[:, 0, 2].view(-1, 1, 1, 1) * attn_1 +
            weights[:, 0, 3].view(-1, 1, 1, 1) * attn_2
        )

        # Channel Mixing
        attn = self.conv3(attn)

        # Convolutional Attention
        x = attn * u

        return x


class CustomMSCAAttention16(CustomMSCAAttention8):
    def __init__(self, channels):
        super().__init__(channels)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Unflatten(1, (1, channels))
        )
        self.weighting = nn.Sequential(
            # nn.Conv1d(1, 1, channels // 4, padding=channels // 8 - 1, stride=channels // 4),
            nn.Linear(channels * 4, channels // 4),
            nn.ReLU(inplace=True),
            # nn.Linear(16 + (channels % 4 != 0), 4),
            nn.Linear(channels // 4, 4),
            nn.Softmax(dim=-1)
        )


    def forward(self, x):
        u = x.clone()

        # Multi-Scale Feature extraction
        attn0 = self.conv0(x)
        attn1 = self.conv1(attn0)
        sum1 = attn1 + attn0
        attn2 = self.conv2(sum1)
        sum2 = sum1 + attn2
        attn3 = self.conv3(sum2)
        sum3 = sum2 + attn3
        attn4 = self.conv4(sum3)

        pooled = torch.cat(
            [
                self.gap(attn1),
                self.gap(attn2),
                self.gap(attn3),
                self.gap(attn4)
            ],
            dim=2
        )
        weights = self.weighting(pooled)
        attn = (
            weights[:, 0, 0].view(-1, 1, 1, 1) * attn1 +
            weights[:, 0, 1].view(-1, 1, 1, 1) * attn2 +
            weights[:, 0, 2].view(-1, 1, 1, 1) * attn3 +
            weights[:, 0, 3].view(-1, 1, 1, 1) * attn4
        )
        attn = self.channel_mixing(attn)

        # Convolutional Attention
        out = attn * u

        return out


class CustomMSCAAttention17(CustomMSCAAttention2):
    def __init__(self, channels):
        super().__init__(channels)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Unflatten(1, (channels, ))
        )
        self.weighting = nn.Sequential(
            nn.Linear(channels * 4, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 4),
            nn.Softmax(dim=-1)
        )


    def forward(self, x):
        u = x.clone()

        # Multi-Scale Feature extraction
        attn0 = self.conv0(x)
        attn1 = self.conv1(attn0)
        sum1 = attn1 + attn0
        attn2 = self.conv2(sum1)
        sum2 = sum1 + attn2
        attn3 = self.conv3(sum2)
        sum3 = sum2 + attn3
        attn4 = self.conv4(sum3)

        pooled = torch.cat(
            [
                self.gap(attn1),
                self.gap(attn2),
                self.gap(attn3),
                self.gap(attn4)
            ],
            dim=1
        )
        weights = self.weighting(pooled)
        attn = (
            weights[:, 0].view(-1, 1, 1, 1) * attn1 +
            weights[:, 1].view(-1, 1, 1, 1) * attn2 +
            weights[:, 2].view(-1, 1, 1, 1) * attn3 +
            weights[:, 3].view(-1, 1, 1, 1) * attn4
        )
        attn = self.channel_mixing(attn)

        # Convolutional Attention
        out = attn * u

        return out


class MSCASpatialAttention(BaseModule):
    """Spatial Attention Module in Multi-Scale Convolutional Attention Module
    (MSCA).

    Args:
        in_channels (int): The dimension of channels.
        attention_kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        attention_kernel_paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
    """

    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU')):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        # print("MSCASpatialAttention", end=', ')
        # print("in channels", in_channels, end=', ')
        # print("hidden channels", hidden_channels)
        self.proj_1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.activation = build_activation_layer(act_cfg)
        self.spatial_gating_unit = MSCAAttention(hidden_channels,
                                                 attention_kernel_sizes,
                                                 attention_kernel_paddings)
        self.proj_2 = nn.Conv2d(hidden_channels, in_channels, 1)

    def forward(self, x):
        """Forward function."""

        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class CustomMSCASpatialAttention(MSCASpatialAttention):
    def __init__(self,
                 custom_version,
                 in_channels,
                 hidden_channels,
                 **kwargs):
        super().__init__(in_channels, hidden_channels, **kwargs)

        match custom_version:
            case 1:
                self.spatial_gating_unit = CustomMSCAAttention(hidden_channels)
            case 2:
                self.spatial_gating_unit = CustomMSCAAttention2(hidden_channels)
            case 3:
                self.spatial_gating_unit = CustomMSCAAttention3(hidden_channels)
            case 4:
                self.spatial_gating_unit = CustomMSCAAttention4(hidden_channels)
            case 5:
                self.spatial_gating_unit = CustomMSCAAttention5(hidden_channels)
            case 6:
                self.spatial_gating_unit = CustomMSCAAttention6(hidden_channels)
            case 7:
                self.spatial_gating_unit = CustomMSCAAttention7(hidden_channels)
            case 8:
                self.spatial_gating_unit = CustomMSCAAttention8(hidden_channels)
            case 9:
                self.spatial_gating_unit = CustomMSCAAttention9(hidden_channels)
            case 10:
                self.spatial_gating_unit = CustomMSCAAttention10(hidden_channels)
            case 11:
                self.spatial_gating_unit = CustomMSCAAttention11(hidden_channels)
            case 12:
                self.spatial_gating_unit = CustomMSCAAttention12(hidden_channels)
            case 13:
                self.spatial_gating_unit = CustomMSCAAttention13(hidden_channels)
            case 15:
                self.spatial_gating_unit = CustomMSCAAttention15(hidden_channels)
            case 16:
                self.spatial_gating_unit = CustomMSCAAttention16(hidden_channels)
            case 17:
                self.spatial_gating_unit = CustomMSCAAttention17(hidden_channels)


class AttentionModule(BaseModule):
    """Spatial Attention Module in Multi-Scale Convolutional Attention Module
    (MSCA).

    Args:
        in_channels (int): The dimension of channels.
        attention_kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        attention_kernel_paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
    """

    def __init__(self,
                 in_channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU'),
                 is_ham=False,ham_kwargs=dict(), ham_norm_cfg=None):
        super().__init__()
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.activation = build_activation_layer(act_cfg)
        if is_ham:
            self.spatial_gating_unit  = Hamburger(in_channels, ham_kwargs, ham_norm_cfg)
        else:
            self.spatial_gating_unit = MSCAAttention(in_channels,
                                                 attention_kernel_sizes,
                                                 attention_kernel_paddings)
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        """Forward function."""

        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        # print('========\nbefore spatial gating unit', x.shape)
        x = self.spatial_gating_unit(x)
        # print('after spatial gating unit', x.shape)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class MSCABlock(BaseModule):
    """Basic Multi-Scale Convolutional Attention Block. It leverage the large-
    kernel attention (LKA) mechanism to build both channel and spatial
    attention. In each branch, it uses two depth-wise strip convolutions to
    approximate standard depth-wise convolutions with large kernels. The kernel
    size for each branch is set to 7, 11, and 21, respectively.

    Args:
        channels (int): The dimension of channels.
        attention_kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        attention_kernel_paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
        mlp_ratio (float): The ratio of multiple input dimension to
            calculate hidden feature in MLP layer. Defaults: 4.0.
        drop (float): The number of dropout rate in MLP block.
            Defaults: 0.0.
        drop_path (float): The ratio of drop paths.
            Defaults: 0.0.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 channels,
                 hidden_channels=None,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 mlp_channel_attention_type=None):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, channels)[1]
        self.attn = MSCASpatialAttention(channels, channels, attention_kernel_sizes,
                                         attention_kernel_paddings, act_cfg)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, channels)[1]
        mlp_hidden_channels = int(channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_channels,
            act_cfg=act_cfg,
            drop=drop,
            channel_attention_type=mlp_channel_attention_type
        )
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)

    def forward(self, x, H, W):
        """Forward function."""

        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class MSCABlockWithCustomSpatialAttention(MSCABlock):
    def __init__(self, custom_version=1, **kwargs):
        super().__init__(**kwargs)
        self.attn = CustomMSCASpatialAttention(
            custom_version,
            kwargs['channels'],
            kwargs['hidden_channels'],
            act_cfg=kwargs['act_cfg']
        )


class MSCABlockWithHam(BaseModule):
    def __init__(self,
                 channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 is_ham=False, ham_kwargs=dict(), ham_norm_cfg=None):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, channels)[1]
        self.attn = AttentionModule(channels, attention_kernel_sizes,
                                    attention_kernel_paddings, act_cfg,
                                    is_ham, ham_kwargs, ham_norm_cfg)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, channels)[1]
        mlp_hidden_channels = int(channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_channels,
            act_cfg=act_cfg,
            drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)

    def forward(self, x, H, W):
        """Forward function."""

        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool1d(x, output_size=1)
        avg = F.adaptive_avg_pool1d(x, output_size=1)
        b, c, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output)
        output = output * x
        return output


class SqEx(nn.Module):
    def __init__(self, channels, hidden_channels=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1)
        )

    def forward(self, x):
        scale = self.mlp(x)
        return scale * x


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class MSCABlockWithChannelAttention(BaseModule):
    def __init__(self,
                 channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 channel_attn = 'SE',
                 ham_kwargs=dict(), ham_norm_cfg=None,
                input_size=256):
        super().__init__()
        self.norm0 = build_norm_layer(dict(type='BN1d', requires_grad=True), channels, channels)[1]

        self.warmup_progress = 0.
        self.channel_attention_type = channel_attn
        match channel_attn:
            case 'Ham':
                self.channel_attention = CustomHamburger(channels, 64, ham_kwargs, ham_norm_cfg)
            case 'CBAM':
                self.channel_attention = CAM(channels, r=1)
            case 'SA':
                self.input_size = input_size
                self.reduce = nn.Linear(self.input_size ** 2, 64)
                self.channel_attention = nn.MultiheadAttention(64, num_heads=8, batch_first=True)
                self.expand = nn.Linear(64, self.input_size ** 2)
            case 'SE':
                self.channel_attention = SqueezeExcitation(channels, channels // 16)
                # self.channel_attention = SqEx(channels, channels // 16)
            case 'ECA':
                t = (math.log2(channels) + 1) // 2
                k = t if t % 2 else t + 1
                self.channel_attention = eca_layer(k_size=int(k))
            case _:
                self.channel_attention = nn.Identity()


        self.norm1 = build_norm_layer(norm_cfg, channels)[1]
        self.attn = MSCASpatialAttention(channels, channels, attention_kernel_sizes,
                                         attention_kernel_paddings, act_cfg)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, channels)[1]
        mlp_hidden_channels = int(channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_channels,
            act_cfg=act_cfg,
            drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_0 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)

    def forward(self, x, H, W):
        """Forward function."""

        B, N, C = x.shape
        x = x.permute(0, 2, 1)

        x = x.view(B, C, N)
        if self.channel_attention_type in ['SE', 'ECA']:
            normed_x = self.norm0(x)
            channel_attn = self.drop_path(self.layer_scale_0.unsqueeze(-1)
                *
                self.channel_attention(normed_x.view(B, C, H, W)).view(B, C, N))
        # elif self.channel_attention_type == 'SA':
        #     x = x
        #     normed_x = self.norm0(x).view(B, C, H, W)
        #     padding = (0, self.input_size - W, 0, self.input_size - H)
        #     normed_x = F.pad(normed_x, padding, mode='constant', value=0).view(B, C, self.input_size ** 2)
        #
        #     low_dim_x = self.reduce(normed_x)
        #     attn_output = self.channel_attention(low_dim_x, low_dim_x, low_dim_x, need_weights=False)[0]
        #     attn_output = self.expand(attn_output)
        #
        #     attn_output = attn_output.view(B, C, self.input_size, self.input_size)[:, :, :H, :W]
        #     x = x.view(B, C, H, W) + self.drop_path(
            #     self.layer_scale_0.unsqueeze(-1).unsqueeze(-1) * attn_output
            # )
        else:
            normed_x = self.norm0(x)
            channel_attn = self.drop_path(self.layer_scale_0.unsqueeze(-1)
                *
                self.channel_attention(normed_x))
        x = (2 - self.warmup_progress) * x + self.warmup_progress * channel_attn
        # print("===self.alpha", self.alpha)

        x = x.view(B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))

        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(BaseModule):
    """Image to Patch Embedding.

    Args:
        patch_size (int): The patch size.
            Defaults: 7.
        stride (int): Stride of the convolutional layer.
            Default: 4.
        in_channels (int): The number of input channels.
            Defaults: 3.
        embed_dims (int): The dimensions of embedding.
            Defaults: 768.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_channels=3,
                 embed_dim=768,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2)
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        """Forward function."""

        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W



@MODELS.register_module()
class MSCANPretrain(BaseModule):
    """SegNeXt Multi-Scale Convolutional Attention Network (MCSAN) backbone.

    This backbone is the implementation of `SegNeXt: Rethinking
    Convolutional Attention Design for Semantic
    Segmentation <https://arxiv.org/abs/2209.08575>`_.
    Inspiration from https://github.com/visual-attention-network/segnext.

    Args:
        in_channels (int): The number of input channels. Defaults: 3.
        embed_dims (list[int]): Embedding dimension.
            Defaults: [64, 128, 256, 512].
        mlp_ratios (list[int]): Ratio of mlp hidden dim to embedding dim.
            Defaults: [4, 4, 4, 4].
        drop_rate (float): Dropout rate. Defaults: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.
        depths (list[int]): Depths of each Swin Transformer stage.
            Default: [3, 4, 6, 3].
        num_stages (int): MSCAN stages. Default: 4.
        attention_kernel_sizes (list): Size of attention kernel in
            Attention Module (Figure 2(b) of original paper).
            Defaults: [5, [1, 7], [1, 11], [1, 21]].
        attention_kernel_paddings (list): Size of attention paddings
            in Attention Module (Figure 2(b) of original paper).
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
        norm_cfg (dict): Config of norm layers.
            Defaults: dict(type='SyncBN', requires_grad=True).
        pretrained (str, optional): model pretrained path.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None,
                 mlp_channel_attention_type=None):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_channels=in_channels if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    norm_cfg=norm_cfg)

            block = nn.ModuleList([
                MSCABlock(
                    channels=embed_dims[i],
                    attention_kernel_sizes=attention_kernel_sizes,
                    attention_kernel_paddings=attention_kernel_paddings,
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    mlp_channel_attention_type=mlp_channel_attention_type
                ) for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

    def init_weights(self):
        """Initialize modules of MSCAN."""

        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):
        """Forward function."""

        # print('shape ', x.shape)
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        # return outs
        return (x,)

