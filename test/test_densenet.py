import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import CIFAR10

from densenet import DenseNet, TransitionLayer


def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train = CIFAR10('./CIFAR10', train=True, transform=transform, download=True)
    test = CIFAR10('./CIFAR10', train=False, transform=transform, download=True)

    train_x = train.train_data.astype('float32')
    train_y = np.array(train.train_labels)
    test_x = test.test_data.astype('float32')
    test_y = np.array(test.test_labels)
    return train_x, train_y, test_x, test_y


def convert(x):
    return x.reshape(-1, 3, 32, 32)


def test_hello():
    train_x, train_y, test_x, test_y = get_data()
    x = Variable(torch.FloatTensor(convert(train_x[0:1])))
    model: nn.Module = DenseNet(24, blocks=[3, 3, 3])  # TransitionLayer(in_filter=3, out_filter=3)
    pred = model(x)
    print(pred.size(), x.size())
