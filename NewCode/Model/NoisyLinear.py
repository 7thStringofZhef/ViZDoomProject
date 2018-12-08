import torch
from torch import nn
from torch.nn import functional
import math

# Noisy linear layer as in Rainbow
class NoisyLinearLayer(nn.Module):
    def __init__(self, inputs, outputs, initStd):
        super(NoisyLinearLayer, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.initStd = initStd
        self.weightMean = nn.Parameter(torch.empty(outputs, inputs))
        self.weightSigma = nn.Parameter(torch.empty(outputs, inputs))
        self.register_buffer('weightEpsilon', torch.empty(outputs, inputs))
        self.biasMean = nn.Parameter(torch.empty(outputs))
        self.biasSigma = nn.Parameter(torch.empty(outputs))
        self.register_buffer('biasEpsilon', torch.empty(outputs))
        self.resetParameters()
        self.sample_noise()

    # Reset weigjts
    def resetParameters(self):
        meanRange = 1.0 / math.sqrt(self.inputs)
        self.weightMean.data.uniform_(-meanRange, meanRange)
        self.weightSigma.data.fill_(self.initStd / math.sqrt(self.inputs))
        self.biasMean.data.uniform_(-meanRange, meanRange)
        self.biasSigma.data.fill_(self.initStd / math.sqrt(self.outputs))

    # Scale random noise to size of input/outputs
    def scaleNoise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    # Reset noise
    def sample_noise(self):
        epsIn = self.scaleNoise(self.inputs)
        epsOut = self.scaleNoise(self.outputs)
        self.weightEpsilon.copy_(epsOut.ger(epsIn))
        self.biasEpsilon.copy_(epsOut)

    # Define forward pass
    # If training, we inject noise. If testing, we leave it out
    def forward(self, stateInput):
        if self.training:
            return functional.linear(stateInput, self.weightMean+self.weightSigma*self.weightEpsilon,
                                     self.biasMean+self.biasSigma*self.biasEpsilon)
        else:
            return functional.linear(stateInput, self.weightMean, self.biasMean)
