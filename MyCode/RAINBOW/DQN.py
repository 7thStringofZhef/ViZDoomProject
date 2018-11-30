import torch.nn as nn
from torch.autograd import Variable as V
import torch

import numpy as np

#Basic DQN architecture
class DQNModule(nn.Module):
    def __init__(self, inputShape=(3,60,108), gameVars=[('health', 101), ('AMMO2', 301)], gameVarBucketSizes=[10,1]):
        super(DQNModule, self).__init__()

        buildCNNComponent(self, inputShape)
        self.outputDim = self.convolutionalOutputDim

        buildGameVariableComponent(self, gameVars, gameVarBucketSizes)


# CNN component of DQN (based on Arnold)
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

# Game variables component of DQN (e.g., health and ammo)
class GameVariableEmbedding(nn.Embedding):
    def __init__(self, bucketSize, numEmbeddings, *args, **kwargs):
        self.bucketSize = bucketSize
        trueNumEmbeddings = (numEmbeddings+bucketSize-1) // bucketSize
        super(GameVariableEmbedding, self).__init__(trueNumEmbeddings, *args, **kwargs)

    def forward(self, indices):
        return super(GameVariableEmbedding, self).forward(indices.div(self.bucketSize))

def buildGameVariableComponent(module, gameVars, gameVarBucketSizes):
    module.gameVariables = gameVars
    module.numGameVariables = len(gameVars)
    module.gameVariableEmbeddings = []
    for i, (name, numValues) in enumerate(gameVars):
        embedding = GameVariableEmbedding(gameVarBucketSizes[i], numValues)
        setattr(module, '%s_emb' % name, embedding)
        module.gameVariableEmbeddings.append(embedding)


def buildRecurrentLayer(module, inputShape):
    module = nn.LSTM(512)