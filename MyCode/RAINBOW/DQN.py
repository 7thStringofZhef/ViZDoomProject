import torch.nn as nn
from torch.autograd import Variable as V
import torch

import numpy as np
from ..DOOM.Params import doomParams


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

    def __init__(self, params=doomParams):
        super(DQNModuleBase, self).__init__()

        self.numActions = params.numActions
        self.hiddenDimension = params.hiddenDimensions
        # build CNN network
        buildCNNComponent(self, params.inputShape)
        self.outputDim = self.convolutionalOutputDim

        # game variables network
        buildGameVariableComponent(self, params.gameVariables, params.gameVariableBucketSizes)
        self.outputDim += sum(params.embeddingDim)

        # dropout layer. Is it used?
        #self.dropout_layer = nn.Dropout(params.dropout)

        # Estimate state-action value function Q(s, a)
        # If dueling network, estimate advantage function A(s, a)
        self.Q = nn.Linear(params.hiddenDimensions, self.numActions)

        self.dueling = params.dueling
        if self.dueling:
            self.V = nn.Linear(params.hiddenDimensions, 1)

    def base_forward(self, inputBuffers, inputVariables):
        """
        Argument sizes:
            - inputBuffers of shape (batch_size, conv_input_size, h, w)
            - inputScreens of shape (batch_size,)
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
        batch_size = inputBuffers.size(0)

        # convolution
        inputBuffers = inputBuffers / 255.
        convOutput = self.convolutional(inputBuffers).view(batch_size, -1)

        # game variables
        embeddings = [self.gameVariableEmbeddings[i](inputVariables[i])
                        for i in range(self.n_variables)]

        # create state input
        output = torch.cat([convOutput] + embeddings, 1)

        return output

    def head_forward(self, stateInput):
        if self.dueling:
            a = self.Q(stateInput)  # advantage branch
            v = self.V(stateInput)   # state value branch
            a -= a.mean(1, keepdim=True).expand(a.size())
            return v.expand(a.size()) + a
        else:
            return self.Q(stateInput)


class DQNModuleRecurrent(DQNModuleBase):

    def __init__(self, params):
        super(DQNModuleRecurrent, self).__init__(params)
        self.rnn = nn.LSTM(self.outputDim, params.hiddenDimensions,
                           1,
                           batch_first=True)

    def forward(self, inputScreens, inputVariables, prev_state):
        """
        Argument sizes:
            - x_screens of shape (batch_size, seq_len, n_fm, h, w)
            - x_variables list of n_var tensors of shape (batch_size, seq_len)
        """
        batchSize = inputScreens.size(0)
        seqLength = inputScreens.size(1)

        # We're doing a batched forward through the network base
        # Flattening seq_len into batch_size ensures that it will be applied
        # to all timesteps independently.
        state_input = self.base_forward(
            inputScreens.view(batchSize * seqLength, *inputScreens.size()[2:]),
            [v.contiguous().view(batchSize * seqLength) for v in inputVariables]
        )

        # unflatten the input and apply the RNN
        rnn_input = state_input.view(batchSize, seqLength, self.outputDim)
        rnn_output, next_state = self.rnn(rnn_input, prev_state)
        rnn_output = rnn_output.contiguous()

        # apply the head to RNN hidden states (simulating larger batch again)
        output_sc = self.head_forward(rnn_output.view(-1, self.hiddenDimensionim))

        # unflatten scores and game features
        output_sc = output_sc.view(batchSize, seqLength, output_sc.size(1))

        return output_sc, next_state


class DQN(object):

    def __init__(self, params):
        # network parameters
        self.params = params
        self.screenShape = (params.n_fm, params.height, params.width)
        self.histSize = params.hist_size
        self.n_variables = params.n_variables

        # main module + loss functions
        self.module = self.DQNModuleClass(params)
        self.lossFunction = nn.SmoothL1Loss()  # Huber

        # cuda
        self.cuda = True
        self.module.cuda()

    def get_var(self, x):
        """Move a tensor to a GPU variable."""
        x = torch.autograd.Variable(x)
        return x.cuda()

    def reset(self):
        pass

    def prepare_f_eval_args(self, last_states):
        """
        Prepare inputs for evaluation.
        """
        screens = np.float32([s.screen for s in last_states])
        screens = self.get_var(torch.FloatTensor(screens))

        variables = np.int64([s.variables for s in last_states])
        variables = self.get_var(torch.LongTensor(variables))

        return screens, variables

    def prepare_f_train_args(self, screens, variables,
                             actions, rewards, isDone):
        """
        Prepare inputs for training.
        """
        # convert tensors to torch Variables
        # TODO remove the .copy below
        screens = self.get_var(torch.FloatTensor(np.float32(screens).copy()))
        variables = self.get_var(torch.LongTensor(np.int64(variables).copy()))
        rewards = self.get_var(torch.FloatTensor(np.float32(rewards).copy()))
        isDone = self.get_var(torch.FloatTensor(np.float32(isDone).copy()))

        return screens, variables, actions, rewards, isDone

    def next_action(self, last_states):
        scores, pred_features = self.f_eval(last_states)
        scores = scores[0, -1]
        action_id = scores.data.max(0)[1][0]
        self.pred_features = pred_features
        return action_id

class DQNRecurrent(DQN):

    DQNModuleClass = DQNModuleRecurrent

    def __init__(self, params=doomParams):
        super(DQNRecurrent, self).__init__(params)
        h_0 = torch.FloatTensor(1, params.batchSize,
                                params.hiddenDimensions).zero_()
        self.init_state_t = self.get_var(h_0)
        self.init_state_e = torch.autograd.Variable(self.init_state_t[:, :1, :].data.clone(), volatile=True)
        self.init_state_t = (self.init_state_t, self.init_state_t)
        self.init_state_e = (self.init_state_e, self.init_state_e)
        self.reset()

    def reset(self):
        # prev_state is only used for evaluation, so has a batch size of 1
        self.prev_state = self.init_state_e

    def f_eval(self, last_states):

        screens, variables = self.prepare_f_eval_args(last_states)

        # feed the last `hist_size` ones
        output = self.module(
            screens.view(1, self.histSize, *self.screenShape),
            [variables[:, i].contiguous().view(1, self.histSize)
             for i in range(self.params.n_variables)],
            prev_state=self.prev_state
        )

        # do not return the recurrent state
        return output[:-1]

    def f_train(self, screens, variables, actions, rewards, isDone):

        screens, variables, actions, rewards, isDone = \
            self.prepare_f_train_args(screens, variables, actions, rewards, isDone)

        batchSize = self.params.batchSize
        seqLength = self.histSize + self.params.numRecurrentUpdates

        output = self.module(
            screens,
            [variables[:, :, i] for i in range(self.params.numGameVariables)],
            prev_state=self.init_state_t
        )[0]

        # compute scores
        mask = torch.ByteTensor(output.size()).fill_(0)
        for i in range(batchSize):
            for j in range(seqLength - 1):
                mask[i, j, int(actions[i, j])] = 1
        scores1 = output.masked_select(self.get_var(mask))
        scores2 = rewards + (
            self.params.gamma * output[:, 1:, :].max(2)[0] * (1 - isDone)
        )

        # dqn loss
        loss = self.lossFunction(
            scores1.view(batchSize, -1)[:, -self.params.numRecurrentUpdates:],
            torch.autograd.Variable(scores2.data[:, -self.params.numRecurrentUpdates:])
        )

        return loss