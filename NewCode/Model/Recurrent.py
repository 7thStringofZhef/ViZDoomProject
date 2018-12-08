import torch
import torch.nn as nn
import torch.nn.functional as F
from .NoisyLinear import NoisyLinearLayer
from .Feedforward import DoomConvolutionalBody

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DRQN(nn.Module):
    def __init__(self, params, body=DoomConvolutionalBody):
        super(DRQN, self).__init__()

        self.input_shape = params.inputShape
        self.num_actions = params.numActions
        self.noisy = params.noisy
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
        x = self.fc2(out)

        return x, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1 * self.num_directions, batch_size, self.hiddenDimensions, device=device, dtype=torch.float)

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc2.sample_noise()

class DuelingDRQN(nn.Module):
    def __init__(self, params, body=DoomConvolutionalBody):
        super(DRQN, self).__init__()

        self.input_shape = params.inputShape
        self.num_actions = params.numActions
        self.noisy = params.noisy
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
        x = self.fc2(out)

        return x, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1 * self.num_directions, batch_size, self.hiddenDimensions, device=device, dtype=torch.float)

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc2.sample_noise()