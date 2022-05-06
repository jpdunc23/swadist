"""ResNet
Adapted from:
 - https://d2l.ai/chapter_convolutional-modern/resnet.html
 - https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""

from copy import deepcopy

import numpy as np
import torch
from torch import nn


__all__ = [
    "ResNet",
]


def conv3x3(in_channels, out_channels, batch_norm=True, relu=True, stride=1):
    """3x3 convolution, batch normalization, relu."""
    mods = []
    mods.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=stride, padding=1, bias=False))
    if batch_norm:
        mods.append(nn.BatchNorm2d(out_channels))
    if relu:
        mods.append(nn.ReLU(inplace=True))
    return nn.Sequential(*mods)


class ResidualBlock(nn.Module):
    """The basic residual block of ResNet."""
    def __init__(self, in_channels, out_channels, batch_norm=True, stride=1):
        super().__init__()

        # convolutions
        self.conv1 = conv3x3(in_channels, out_channels, batch_norm, True, stride)
        self.conv2 = conv3x3(out_channels, out_channels, batch_norm, False)

        # downsampling
        if stride != 1 or in_channels != out_channels:
            # TODO: when batch_norm is False, use manual downsampling
            mods = []
            mods.append(nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  stride=stride, bias=False))
            if batch_norm:
                mods.append(nn.BatchNorm2d(out_channels))
            self.downsample = nn.Sequential(*mods)
        else:
            self.downsample = None

        # final ReLU
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            identity = self.downsample(x)

        # don't do in-place addition after ReLU
        # https://github.com/pytorch/pytorch/issues/5687#issuecomment-412681482
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Parameters
    ----------
    n_classes: int
        The number of output classes.
    in_channels: int, optional
        Number of input channels.
    in_kernel_size: int, optional
        Size of the kernel for the input in the first convolutional layer.
    stack_sizes: List[int], optional
        The number of residual blocks in each group of layers where feature map sizes stay constant.
        The default gives ResNet-18 (or ResNet-14 when `in_kernel_size` is 3).
    batch_norm: bool, optional
        If True, uses batch normalization at the end of each convolutional layer.
    """
    def __init__(self, n_classes, in_channels=3, in_kernel_size=7, stack_sizes=None,
                 batch_norm=True, device=None):
        super().__init__()

        if stack_sizes is None:
            stack_sizes = [2, 2, 2, 2]
        if in_kernel_size not in [3, 7]:
            raise ValueError("The first layer's kernel size must be 3 or 7.")
        if in_kernel_size == 7:
            base_channels = 64
        else:
            base_channels = 16
        self.convs = ['layer1', 'stack1', 'stack2', 'stack3']
        self.layer1 = self._resnet_first_layer(in_channels, base_channels, in_kernel_size,
                                               batch_norm)
        self.stack1 = self._resnet_stack(base_channels, base_channels, stack_sizes[0],
                                         batch_norm, first_stack=True)
        self.stack2 = self._resnet_stack(base_channels, base_channels*2, stack_sizes[1],
                                         batch_norm)
        self.stack3 = self._resnet_stack(base_channels*2, base_channels*4, stack_sizes[2],
                                         batch_norm)
        if in_kernel_size == 7:
            out_channels = base_channels*8
            self.stack4 = self._resnet_stack(base_channels*4, out_channels, stack_sizes[3],
                                             batch_norm)
            self.convs.append('stack4')
        else:
            out_channels = base_channels*4
            self.stack4 = None
        self.final_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, n_classes)
        )

        self.reinitialize_parameters(device=device)


    def reinitialize_parameters(self, device=None):
        if device is not None:
            self.to(device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _resnet_first_layer(self, in_channels, out_channels, kernel_size=7, batch_norm=True):
        mods = []
        stride = 1 if kernel_size != 7 else 2
        padding = 1 if kernel_size != 7 else 3
        mods.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        )
        if batch_norm:
            mods.append(nn.BatchNorm2d(out_channels))
        mods.append(nn.ReLU(inplace=True))
        if kernel_size == 7:
            mods.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return nn.Sequential(*mods)


    def _resnet_stack(self, in_channels, out_channels, n_blocks, batch_norm=True, first_stack=False):
        blocks = []
        for i in range(n_blocks):
            if i == 0 and not first_stack:
                blocks.append(ResidualBlock(in_channels, out_channels, batch_norm, stride=2))
            else:
                blocks.append(ResidualBlock(out_channels, out_channels, batch_norm))
        return nn.Sequential(*blocks)


    def forward(self, x):
        x = self.layer1(x)
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        if self.stack4:
            x = self.stack4(x)
        x = self.final_layer(x)
        return x


    def get_filters(self, w_idx_dict=None):
        """Returns specified or random filters from each convolutional layer.

        Parameters
        ----------

        w_idx_dict: dict (optional)
            Has entries like { 'layer1 conv1': [1,2,3,4] } for each of `self.convs`,
            where the values are lists of ints with positive values no larger than the
            number of channels for the corresponding convolution module.
        """
        with torch.inference_mode():
            filters = {}
            if w_idx_dict is None:
                w_idx_dict = {}
            for conv in self.convs:
                module = self.__getattr__(conv)
                i = 0
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        i += 1
                        if i == 3:
                            break # ignore downsampling
                        key = f'{conv}/conv{i}'
                        filts = m.weight.data.view(-1, 1, *m.kernel_size)
                        if key in w_idx_dict:
                            idxs = w_idx_dict[key]
                        else:
                            # get four random kernels
                            idxs = np.random.choice(len(filts), size=4, replace=False)
                            idxs.sort()
                            w_idx_dict[key] = idxs
                        filters[key] = idxs, filts[idxs]
            return filters, w_idx_dict


    def partial_forward(self, x, stop_module='layer1'):
        with torch.inference_mode():
            for name in self.convs[:self.convs.index(stop_module)]:
                m = self.__getattr__(name)
                x = m(x)
            return x
