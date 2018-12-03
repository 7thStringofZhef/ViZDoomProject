from __future__ import absolute_import

import configparser
import torch
from functools import partial
import numpy as np
import numpy.random as npr
from collections import deque

from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution, GameVariable
from DOOM.FrameProcessing import processImage, gameStateToTensor
from DOOM.Params import doomParams
from RAINBOW.ModelMemory import PrioritizedReplayMemory, LinearSchedule, GameState, blank_trans
from RAINBOW.DQN import DQNRecurrent


def initGameWithParams(configFilePath):
    game = DoomGame()
    game.load_config(configFilePath)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_400X225)
    game.init()
    return game


class DoomGameEnv(object):
    def __init__(self, params=doomParams):
        self.params = params
        self.frameskip = params.frameskip
        self.game = initGameWithParams(params.scenarioPath)
        self.numChannels, self.imageHeight, self.imageWidth = params.inputShape
        self.actions = self.game.get_available_buttons()
        self.numActions = len(self.actions)
        self.actions = np.identity(self.numActions, dtype=np.int32).tolist()  # One-hot encoding of actions
        self.gameVariables = params.gameVariables
        self.numGameVariables = len(self.gameVariables)
        self.reset()

    # Return last history states
    def step(self, action):
        reward = 0
        is_done = False
        self.game.set_action(action)
        for _ in range(self.frameskip):
            reward += self.game.advance_action()
        is_done = self.game.is_episode_finished() or self.game.is_player_dead()
        newState = self.game.get_state()
        processedState = gameStateToTensor(newState)
        self.stateBuffer.append(processedState)
        return list(self.stateBuffer), reward, is_done

    def reset(self):
        self.game.new_episode()
        # Start with a queue of all blank frames
        self.stateBuffer = deque([], maxlen=self.params.recurrenceHistory+1)
        newState = self.game.get_state()
        processedState = gameStateToTensor(newState)
        for i in range(self.params.recurrenceHistory+1):
            self.stateBuffer.append(processedState)
        return list(self.stateBuffer)

    def getEpisodeReward(self):
        return self.game.get_total_reward()


def updateTargetNet(policyNet, targetNet):
    targetNet.module.load_state_dict(policyNet.module.state_dict())

# Perform the update step on my policy network
def optimizeNet(policyNet, targetNet, batch):
    pass

def train(env, params):
    # Policy net
    policyNet = DQNRecurrent(params)
    policyNet.module.train()
    if params.double:
        # Target net
        targetNet = DQNRecurrent(params)
        updateTargetNet(policyNet, targetNet)
        targetNet.module.train()
        for parameter in targetNet.module.parameters():
            parameter.requires_grad = False
    else:
        targetNet = None

    memory = PrioritizedReplayMemory(params)  # Create a memory buffer

    # Anneal beta over time
    if params.prioritizedReplay:
        beta = LinearSchedule(params.priorityBetaSteps, params.priorityBetaEnd, params.priorityBetaStart)
    else:
        beta = None

    # Anneal epsilon over time
    if not params.noisyLinear:
        epsilon = LinearSchedule(params.epsSteps, params.endEps, params.startEps)  # If using noisy linear layer, this doesn't applyS
    else:
        epsilon = None

    # Optimize with Adam
    optimFunction = torch.optim.Adam(policyNet.module.parameters(), lr=params.learningRate)

    frameCounter = 0
    episodeRewards = list()
    episodeLengths = list()


    for episode in range(params.numEpisodes):
        episodeFrameCounter = 0
        isDone = False
        currState = env.reset()
        while not isDone:
            # Take action based on current state or on eps
            if epsilon is not None and npr.randn() < epsilon.value(frameCounter):
                action = npr.randint(0, env.numActions)
            else:
                action = policyNet.next_action(currState)
            pass
            episodeFrameCounter += 1
            frameCounter += 1

        episodeRewards.append(env.getEpisodeReward())
        episodeLengths.append(episodeFrameCounter)








if __name__ == "__main__":
    gameEnv = DoomGameEnv()
    train(gameEnv, doomParams)

