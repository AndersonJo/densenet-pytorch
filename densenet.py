from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn


class TransitionLayer(nn.Sequential):
    def __init__(self, in_filter: int, out_filter: int):
        super(TransitionLayer, self).__init__()

        self.add_module('norm', nn.BatchNorm2d(in_filter))
        # self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_filter, out_channels=out_filter, kernel_size=1, stride=1, padding=0))
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
        self.add_module('norm', nn.BatchNorm2d(in_filter))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_filter, out_filter, kernel_size=1, stride=1, padding=0))


class DenseBlockLayer(nn.Sequential):
    def __init__(self, n_layer: int, bottleneck=True):
        super(DenseBlockLayer, self).__init__()
        for i in range(n_layer):
            if bottleneck:
                # self.add_module(f'bottleneck_{i}', BottleNeckLayer())
                pass


class DenseNet(nn.Module):
    def __init__(self, k: int, blocks: List[int], bottlenecks: List[bool] = None):
        """
        :param k: growth rate. 보통 12, 24, 40 같은 값으로 설정
        :param blocks: DenseBlock안의 layers의 갯수 (ex. blocks=[12, 16, 40])
        """
        super(DenseNet, self).__init__()

        if bottlenecks is None:
            bottlenecks = [True for _ in range(len(blocks))]
        assert len(blocks) == len(bottlenecks)

        # DenseBlock을 타기전 먼저 Convolution을 태워서 channels size를 키운다.
        # 이때 16 또는 growth rate의 두배값으로 결정
        init_filter = 2 * k
        self.net = nn.Sequential(OrderedDict([
            ('init_conv', nn.Conv2d(in_channels=3, out_channels=init_filter,
                                    kernel_size=3, stride=1, padding=1, bias=False))
        ]))

        for i, (n_layer, bottleneck) in enumerate(zip(blocks, bottlenecks)):
            dense_block = DenseBlockLayer(n_layer, bottleneck)

    def forward(self, x):
        h = self.net(x)
        return h
