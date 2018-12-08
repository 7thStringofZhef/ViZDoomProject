import torch
import torch.nn as nn
import torch.nn.functional as F
from .NoisyLinear import NoisyLinearLayer
from .Feedforward import DoomConvolutionalBody

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DRQN(nn.Module):
    def __init__(self, params, body=DoomConvolutionalBody):
        super(DRQN, self).__init__()

        self.inputShape = params.inputShape
        self.numActions = params.numActions
        self.noisy = params.noisyLinear
        self.hiddenDimensions = params.hiddenDimensions

        self.body = body(self.inputShape)
        self.gru = nn.GRU(self.body.feature_size(), params.hiddenDimensions, num_layers=1, batch_first=True,
                          bidirectional=False)
        self.fc2 = nn.Linear(params.hiddenDimensions, self.numActions) if not self.noisy \
            else NoisyLinearLayer(params.hiddenDimensions, self.numActions, params.noisyParam)

    def forward(self, inputSeq, hx=None):
        batchSize = inputSeq.size(0)
        seqLength = inputSeq.size(1)

        inputSeq = inputSeq.view((-1,) + self.inputShape)

        # format outp for batch first gru
        feats = self.body(inputSeq).view(batchSize, seqLength, -1)
        hidden = self.init_hidden(batchSize) if hx is None else hx
        out, hidden = self.gru(feats, hidden)
        inputSeq = self.fc2(out)

        return inputSeq, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hiddenDimensions, device=device, dtype=torch.float)

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc2.sample_noise()

class DuelingDRQN(nn.Module):
    def __init__(self, params, body=DoomConvolutionalBody):
        super(DuelingDRQN, self).__init__()

        self.inputShape = params.inputShape
        self.numActions = params.numActions
        self.noisy = params.noisyLinear
        self.hiddenDimensions = params.hiddenDimensions

        self.body = body(self.inputShape)
        self.gru = nn.GRU(self.body.feature_size(), params.hiddenDimensions, num_layers=1, batch_first=True,
                          bidirectional=False)

        self.adv2 = nn.Linear(params.hiddenDimensions, self.numActions) if not self.noisy \
            else NoisyLinearLayer(params.hiddenDimensions, self.numActions, params.noisyParam)
        self.val2 = nn.Linear(params.hiddenDimensions, 1) if not self.noisy \
            else NoisyLinearLayer(params.hiddenDimensions, 1, params.noisyParam)

    def forward(self, inputSeq, hx=None):
        batchSize = inputSeq.size(0)
        seqLength = inputSeq.size(1)

        inputSeq = inputSeq.view((-1,) + self.inputShape)

        # format outp for batch first gru
        feats = self.body(inputSeq).view(batchSize, seqLength, -1)
        hidden = self.init_hidden(batchSize) if hx is None else hx
        out, hidden = self.gru(feats, hidden)
        adv = self.adv2(inputSeq)
        val = self.val2(inputSeq)

        return adv + val - adv.mean(), hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hiddenDimensions, device=device, dtype=torch.float)

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.adv2.sample_noise()
            self.val2.sample_noise()

class CategoricalDRQN(nn.Module):
    def __init__(self, params, body=DoomConvolutionalBody):
        super(CategoricalDRQN, self).__init__()

        self.inputShape = params.inputShape
        self.numActions = params.numActions
        self.noisy = params.noisyLinear
        self.hiddenDimensions = params.hiddenDimensions
        self.atoms = params.atoms

        self.body = body(self.inputShape)
        self.gru = nn.GRU(self.body.feature_size(), params.hiddenDimensions, num_layers=1, batch_first=True,
                          bidirectional=False)

        self.fc2 = nn.Linear(params.hiddenDimensions, self.numActions) if not self.noisy \
            else NoisyLinearLayer(params.hiddenDimensions, self.numActions, params.noisyParam)

    def forward(self, inputSeq, hx=None):
        batchSize = inputSeq.size(0)
        seqLength = inputSeq.size(1)

        inputSeq = inputSeq.view((-1,) + self.inputShape)

        # format outp for batch first gru
        feats = self.body(inputSeq).view(batchSize, seqLength, -1)
        hidden = self.init_hidden(batchSize) if hx is None else hx
        out, hidden = self.gru(feats, hidden)
        inputSeq = self.fc2(out)
        return inputSeq, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hiddenDimensions, device=device, dtype=torch.float)

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc2.sample_noise()

class CategoricalDuelingDRQN(nn.Module):
    def __init__(self, params, body=DoomConvolutionalBody):
        super(CategoricalDuelingDRQN, self).__init__()

        self.inputShape = params.inputShape
        self.numActions = params.numActions
        self.noisy = params.noisyLinear
        self.hiddenDimensions = params.hiddenDimensions
        self.atoms = params.atoms

        self.body = body(self.inputShape)
        self.gru = nn.GRU(self.body.feature_size(), params.hiddenDimensions, num_layers=1, batch_first=True,
                          bidirectional=False)

        self.adv2 = nn.Linear(params.hiddenDimensions, self.numActions*self.atoms) if not self.noisy \
            else NoisyLinearLayer(params.hiddenDimensions, self.numActions*self.atoms, params.noisyParam)
        self.val2 = nn.Linear(params.hiddenDimensions, 1*self.atoms) if not self.noisy \
            else NoisyLinearLayer(params.hiddenDimensions, 1*self.atoms, params.noisyParam)

    def forward(self, inputSeq, hx=None):
        batchSize = inputSeq.size(0)
        seqLength = inputSeq.size(1)

        inputSeq = inputSeq.view((-1,) + self.inputShape)

        # format outp for batch first gru
        feats = self.body(inputSeq).view(batchSize, seqLength, -1)
        hidden = self.init_hidden(batchSize) if hx is None else hx
        out, hidden = self.gru(feats, hidden)
        adv = self.adv2(inputSeq)
        val = self.val2(inputSeq)
        final = val + adv - adv.mean(dim=1).view(-1, 1, self.atoms)
        return F.softmax(final, dim=2), hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hiddenDimensions, device=device, dtype=torch.float)

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.adv2.sample_noise()
            self.val2.sample_noise()