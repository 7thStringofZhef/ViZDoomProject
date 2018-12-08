import torch
import torch.nn as nn
import torch.nn.functional as F
from .NoisyLinear import NoisyLinearLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note that for recurrent, inputShape[0] will be histSize*numChannels instead of just numChannels
class DoomConvolutionalBody(nn.Module):
    def __init__(self, inputShape):
        super(DoomConvolutionalBody, self).__init__()

        self.input_shape = inputShape
        self.noisy = False  # No noisy layers regardless

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x

    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)

    def sample_noise(self):
        pass

class DQN(nn.Module):
    def __init__(self, params, body=DoomConvolutionalBody):
        super(DQN, self).__init__()

        self.inputShape = params.inputShape
        self.numActions = params.numActions
        self.noisy = params.noisyLinear

        self.body = body(self.inputShape)

        self.fc1 = nn.Linear(self.body.feature_size(), params.hiddenDimensions) if not self.noisy \
            else NoisyLinearLayer(self.body.feature_size(), params.hiddenDimensions, params.noisyParam)
        self.fc2 = nn.Linear(params.hiddenDimensions, self.numActions) if not self.noisy \
            else NoisyLinearLayer(params.hiddenDimensions, self.numActions, params.noisyParam)

    def forward(self, x):
        x = self.body(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc1.sample_noise()
            self.fc2.sample_noise()


class DuelingDQN(nn.Module):
    def __init__(self, params, body=DoomConvolutionalBody):
        super(DuelingDQN, self).__init__()

        self.inputShape = params.inputShape
        self.num_actions = params.numActions
        self.noisy = params.noisyLinear

        self.body = body(self.inputShape)

        self.adv1 = nn.Linear(self.body.feature_size(), params.hiddenDimensions) if not self.noisy else NoisyLinearLayer(
            self.body.feature_size(), params.hiddenDimensions, params.noisyParam)
        self.adv2 = nn.Linear(params.hiddenDimensions, self.numActions) if not self.noisy \
            else NoisyLinearLayer(params.hiddenDimensions, self.numActions, params.noisyParam)

        self.val1 = nn.Linear(self.body.feature_size(), params.hiddenDimensions) if not self.noisy \
            else NoisyLinearLayer(self.body.feature_size(), params.hiddenDimensions, params.noisyParam)
        self.val2 = nn.Linear(params.hiddenDimensions, 1) if not self.noisy \
            else NoisyLinearLayer(params.hiddenDimensions, 1, params.noisyParam)

    def forward(self, x):
        x = self.body(x)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.adv1.sample_noise()
            self.adv2.sample_noise()
            self.val1.sample_noise()
            self.val2.sample_noise()


class CategoricalDQN(nn.Module):
    def __init__(self, params, body=DoomConvolutionalBody):
        super(CategoricalDQN, self).__init__()

        self.inputShape = params.inputShape
        self.numActions = params.numActions
        self.noisy = params.noisyLinear
        self.atoms = params.atoms

        self.body = body(self.inputShape)

        self.fc1 = nn.Linear(self.body.feature_size(), params.hiddenDimensions) if not self.noisy \
            else NoisyLinearLayer(self.body.feature_size(), params.hiddenDimensions, params.noisyParam)
        self.fc2 = nn.Linear(params.hiddenDimensions, self.numActions * self.atoms) if not self.noisy \
            else NoisyLinearLayer(params.hiddenDimensions, self.numActions * self.atoms, params.noisyParam)

    def forward(self, x):
        x = self.body(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x.view(-1, self.numActions, self.atoms), dim=2)

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc1.sample_noise()
            self.fc2.sample_noise()


class CategoricalDuelingDQN(nn.Module):
    def __init__(self, params, body=DoomConvolutionalBody):
        super(CategoricalDuelingDQN, self).__init__()

        self.inputShape = params.inputShape
        self.numActions = params.numActions
        self.noisy = params.noisyLinear
        self.atoms = params.atoms

        self.body = body(self.inputShape)

        self.adv1 = nn.Linear(self.body.feature_size(), params.hiddenDimensions) if not self.noisy \
            else NoisyLinearLayer(self.body.feature_size(), params.hiddenDimensions, params.noisyParam)
        self.adv2 = nn.Linear(params.hiddenDimensions, self.numActions * self.atoms) if not self.noisy \
            else NoisyLinearLayer(params.hiddenDimensions, self.numActions * self.atoms, params.noisyParam)

        self.val1 = nn.Linear(self.body.feature_size(), params.hiddenDimensions) if not self.noisy \
            else NoisyLinearLayer(self.body.feature_size(), params.hiddenDimensions, params.noisyParam)
        self.val2 = nn.Linear(params.hiddenDimensions, 1 * self.atoms) if not self.noisy \
            else NoisyLinearLayer(params.hiddenDimensions, 1 * self.atoms, params.noisyParam)

    def forward(self, x):
        x = self.body(x)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv).view(-1, self.numActions, self.atoms)

        val = F.relu(self.val1(x))
        val = self.val2(val).view(-1, 1, self.atoms)

        final = val + adv - adv.mean(dim=1).view(-1, 1, self.atoms)

        return F.softmax(final, dim=2)

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.adv1.sample_noise()
            self.adv2.sample_noise()
            self.val1.sample_noise()
            self.val2.sample_noise()