import torch.nn as nn
from torch.autograd import Variable as V
import torch

import numpy as np

#Basic DQN architecture
class DQN(nn.Module):
    def __init__(self, inputShape, numActions):
        super(DQN, self).__init__()
        self.numActions = numActions
        self.inputShape = inputShape
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(inputShape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        convOutputSize = self.getConv(inputShape)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(convOutputSize, 512),
            nn.RelU(),
            nn.Linear(512, numActions)
        )

    # Get the output size of the convolutional layer
    def getConvOutputSize(self, shape):
        output = self.conv(V(torch.zeros(1, *shape)))
        return int(np.prod(output.size()))

    # Run input through the network, return Q values of actions
    def forward(self, input):
        rescaledInput = input.float() / 256
        convOutput = self.conv(rescaledInput).view(rescaledInput.size()[0], -1)
        return self.fc(convOutput)