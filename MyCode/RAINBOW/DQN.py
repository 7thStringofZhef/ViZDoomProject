import torch.nn as nn
from torch.autograd import Variable as V
import torch

import numpy as np

#Basic DQN architecture
class DQN(object):
    def __init__(self, screenShape, historySize):
        self.screenShape = screenShape
        self.histSize = historySize


def buildCNNComponent(module, inputShape):
    channels, height, width = inputShape
    module.convolutional = nn.Sequential([
        nn.Conv2d(channels, 32, (8, 8), stride=(4, 4)),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, (4,4), stride=(2,2)),
        nn.BatchNorm2d(64),
        nn.ReLU,
    ])
    x = torch.autograd.Variable(torch.FloatTensor(1, height, width))
    module.convolutionalOutputDim = module.convolutional(x).nelement()

def buildRecurrentLayer(module, inputShape)