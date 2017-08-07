from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10


class TransitionLayer(nn.Sequential):
    def __init__(self, in_filter: int, out_filter: int):
        super(TransitionLayer, self).__init__()

        self.add_module('norm', nn.BatchNorm2d(in_filter))
        self.add_module('conv', nn.Conv2d(in_filter, out_channels=out_filter, kernel_size=1,
                                          stride=1, padding=0, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2, padding=1))


class BottleNeckLayer(nn.Sequential):
    def __init__(self, in_filter: int, out_filter: int):
        super(BottleNeckLayer, self).__init__()

        self.add_module('norm', nn.BatchNorm2d(in_filter))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_filter, out_filter, kernel_size=1, stride=1, padding=0))


class DenseLayer(nn.Sequential):
    def __init__(self, in_filter: int, out_filter: int):
        super(DenseLayer, self).__init__()
        self.in_filter = in_filter
        self.out_filter = out_filter
        self.add_module('norm', nn.BatchNorm2d(in_filter))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_filter, out_filter, kernel_size=3, stride=1, padding=1))


class DenseBlockLayer(nn.Module):
    def __init__(self, n_layer: int, growth_rate: int, in_filter: int, bottleneck=True):
        super(DenseBlockLayer, self).__init__()
        self.layers = list()

        for i in range(n_layer):

            # growth_rate * (l - 1) + in_filter_size
            n_in = growth_rate * i + in_filter
            n_out = growth_rate * 4

            bottleneck_layer = None
            if bottleneck:
                bottleneck_layer = BottleNeckLayer(n_in, n_out)
                self.add_module(f'bottleneck_{i}', bottleneck_layer)

            dense_layer = DenseLayer(n_out, growth_rate)
            self.add_module(f'dense_layer{i}', dense_layer)
            self.layers.append((bottleneck_layer, dense_layer))

    def forward(self, x):
        h = x
        for bottleneck_layer, dense_layer in self.layers:
            if bottleneck_layer is not None:
                h = bottleneck_layer(h)
            h = dense_layer(h)
            h = torch.cat((x, h), 1)

            x = h
        return h


class DenseNetInit(nn.Module):
    def __init__(self, growth_rate: int):
        """
        :param growth_rate: 보통 12, 24, 40 같은 값으로 설정
        """
        super(DenseNetInit, self).__init__()
        self.growth_rate = growth_rate
        # Init First Convolution which expand the size of filters (channels)
        # DenseBlock을 타기전 먼저 Convolution을 태워서 channels size를 키운다.
        # 이때 16 또는 growth rate의 두배값으로 결정
        self._init_filter = 2 * self.growth_rate

        self.init_conv = nn.Conv2d(in_channels=3, out_channels=self._init_filter,
                                   kernel_size=3, stride=1, padding=1, bias=False)


class DenseNet(DenseNetInit):
    def __init__(self, growth_rate: int, compression_rate: float, n_class: int, fc_size: int,
                 blocks: List[int], bottlenecks: List[bool] = None):
        """
        :param growth_rate: 보통 12, 24, 40 같은 값으로 설정
        :param compression_rate: 1이명 dense layer의 output과 동일하며, 값이 낮을수록 output의 channel의 크기도 줄어든다.
        :param blocks: DenseBlock안의 layers의 갯수 (ex. blocks=[12, 16, 40])
        """
        super(DenseNet, self).__init__(growth_rate)

        # Clean Bottlenecks Parameter
        self.blocks = blocks
        if bottlenecks is None:
            self.bottlenecks = [True for _ in range(len(blocks))]
        else:
            self.bottlenecks = bottlenecks
        assert len(self.blocks) == len(self.bottlenecks)

        # Add DenseBlocks
        self.dense_blocks = list()
        self.transitions = list()
        N = len(self.blocks)
        in_filter = self._init_filter
        for i, (n_layer, bottleneck) in enumerate(zip(self.blocks, self.bottlenecks)):
            # Create Dense Block
            dense_block = DenseBlockLayer(n_layer, self.growth_rate, in_filter, bottleneck)
            self.add_module(f'block{i}', dense_block)
            self.dense_blocks.append(dense_block)

            # Create a Transition
            in_filter += self.growth_rate * n_layer
            out_filter = int(in_filter * compression_rate)
            if i + 1 != N:
                transition_layer = TransitionLayer(in_filter, out_filter)
                self.add_module(f'trainsition{i}', transition_layer)
                self.transitions.append(transition_layer)
                in_filter = out_filter
            else:
                self.transitions.append(None)

        self.init_fc(in_filter, fc_size, n_class)

    def init_fc(self, in_filter: int, fc_size: int, n_class: int):
        # for _ in range(len(self.blocks) - 1):
        #     image_size = self.calculate_conv_output_size(image_size)
        #
        # n_fc = int(image_size[0] * image_size[1] * in_filter)

        self.bn1 = nn.BatchNorm2d(in_filter)
        self.fc1 = nn.Linear(fc_size, n_class)

    def calculate_conv_output_size(self, image_size: Tuple[int, int], filter: int = 2, padding=1,
                                   stride=2):
        image_size = np.array(image_size)
        size = (image_size - filter + 2 * padding) / stride + 1
        size = size.astype('int')
        return size

    def forward(self, x):
        h = self.init_conv(x)
        h = self.forward_dense_layers(h)
        h = self.forward_fully_connected_layer(h)
        return h

    def forward_dense_layers(self, h):
        """
        You can override this function
        """
        for dense_block, transition in zip(self.dense_blocks, self.transitions):
            h = dense_block(h)
            if transition is not None:
                h = transition(h)

        return h

    def forward_fully_connected_layer(self, h):
        h = F.relu(self.bn1(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = h.view(h.size(0), -1)
        h = self.fc1(h)
        return h
