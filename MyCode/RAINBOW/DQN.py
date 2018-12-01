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
    x = torch.autograd.Variable(torch.FloatTensor(1, channels, height, width).zero_())
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


class DQNModuleBase(nn.Module):

    def __init__(self, inputShape=(3,60,108), gameVars=[('health', 101), ('AMMO2', 301)],
                 gameVarBucketSizes=(10,1)):
        super(DQNModuleBase, self).__init__()

        # build CNN network
        buildCNNComponent(self, inputShape)
        self.output_dim = self.convolutionalOutputDim

        # game variables network
        buildGameVariableComponent(self, gameVars, gameVarBucketSizes)
        if self.n_variables:
            self.output_dim += sum(params.variable_dim)

        # dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)

        # game features network
        build_game_features_network(self, params)

        # Estimate state-action value function Q(s, a)
        # If dueling network, estimate advantage function A(s, a)
        self.proj_action_scores = nn.Linear(params.hidden_dim, self.n_actions)

        self.dueling_network = params.dueling_network
        if self.dueling_network:
            self.proj_state_values = nn.Linear(params.hidden_dim, 1)

        # log hidden layer sizes
        logger.info('Conv layer output dim : %i' % self.conv_output_dim)
        logger.info('Hidden layer input dim: %i' % self.output_dim)

    def base_forward(self, x_screens, x_variables):
        """
        Argument sizes:
            - x_screens of shape (batch_size, conv_input_size, h, w)
            - x_variables of shape (batch_size,)
        where for feedforward:
            batch_size == params.batch_size,
            conv_input_size == hist_size * n_feature_maps
        and for recurrent:
            batch_size == params.batch_size * (hist_size + n_rec_updates)
            conv_input_size == n_feature_maps
        Returns:
            - output of shape (batch_size, output_dim)
            - output_gf of shape (batch_size, n_features)
        """
        batch_size = x_screens.size(0)

        # convolution
        x_screens = x_screens / 255.
        conv_output = self.conv(x_screens).view(batch_size, -1)

        # game variables
        if self.n_variables:
            embeddings = [self.game_variable_embeddings[i](x_variables[i])
                          for i in range(self.n_variables)]

        # game features
        if self.n_features:
            output_gf = self.proj_game_features(conv_output)
        else:
            output_gf = None

        # create state input
        if self.n_variables:
            output = torch.cat([conv_output] + embeddings, 1)
        else:
            output = conv_output

        # dropout
        if self.dropout:
            output = self.dropout_layer(output)

        return output, output_gf

    def head_forward(self, state_input):
        if self.dueling_network:
            a = self.proj_action_scores(state_input)  # advantage branch
            v = self.proj_state_values(state_input)   # state value branch
            a -= a.mean(1, keepdim=True).expand(a.size())
            return v.expand(a.size()) + a
        else:
            return self.proj_action_scores(state_input)