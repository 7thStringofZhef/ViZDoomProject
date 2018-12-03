from __future__ import absolute_import

import configparser
import torch
from functools import partial
import numpy as np
import numpy.random as npr
from collections import deque
from time import time

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
        reward += self.game.make_action(action, self.frameskip)
        is_done = self.game.is_episode_finished() or self.game.is_player_dead()
        newState = self.game.get_state()
        if not is_done:
            processedState = gameStateToTensor(newState)
            self.stateBuffer.append(processedState)
        else:
            pass
        return list(self.stateBuffer), reward, is_done

    def reset(self):
        self.game.new_episode()
        # Start with a queue of all blank frames
        self.stateBuffer = deque([], maxlen=self.params.recurrenceHistory+self.params.numRecurrentUpdates)
        newState = self.game.get_state()
        processedState = gameStateToTensor(newState)
        for i in range(self.params.recurrenceHistory+self.params.numRecurrentUpdates):
            self.stateBuffer.append(processedState)
        return list(self.stateBuffer)

    def getEpisodeReward(self):
        return self.game.get_total_reward()


def updateTargetNet(policyNet, targetNet):
    targetNet.module.load_state_dict(policyNet.module.state_dict())

# Perform the update step on my policy network
def optimizeNet(policyNet, targetNet, memory, optimizer, params):
    indices, states, actions, returns, nextStates, isDones, weights = memory.sample(params.batchSize)
    loss = policyNet.f_train(states, actions, returns, isDones, targetNet)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    """
    # If using target Q
    if params.double:
        pass
    else:
        pass

    # If using prioritized replay, update priorities
    if params.prioritizedReplay:
        memory.update
    """


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
    framesBeforeTraining = params.framesBeforeTraining  # Don't start training or annealing before this
    episodeRewards = list()
    episodeLengths = list()
    episodeTimes = list()


    for episode in range(params.numEpisodes):
        print('Starting episode ' + str(episode+1))
        episodeFrameCounter = 0
        isDone = False
        currState = env.reset()
        startTime = time()
        while not isDone:
            # If using noisy linear, reset noise on training frequency
            if params.noisyLinear and frameCounter % params.trainingFrequency == 0:
                policyNet.resetNoise()

            # Take action based on current state or on eps
            if epsilon is not None and npr.rand() < epsilon.value(frameCounter-framesBeforeTraining):
                action = npr.randint(0, env.numActions)
            else:
                action = policyNet.next_action(currState)

            newState, reward, isDone = env.step(oneHotList(action, env.numActions))
            episodeFrameCounter += 1
            frameCounter += 1
            memory.append(currState[-1], action, reward, isDone)  # Add last frame of game state
            # Populate with random experiences first
            if frameCounter > 4000:
                # If it's time to train
                if frameCounter % params.trainingFrequency == 0:
                    optimizeNet(policyNet, targetNet, memory, optimFunction, params)

                # If it's time to update target net
                if params.double and frameCounter % params.targetUpdateFrequency == 0:
                    updateTargetNet(policyNet, targetNet)




        print('Episode reward ' + str(env.getEpisodeReward()))
        episodeRewards.append(env.getEpisodeReward())
        episodeLengths.append(episodeFrameCounter)
        episodeTimes.append(time()-startTime)

    return episodeRewards, episodeLengths, episodeTimes



def oneHotList(action, numActions):
    oneHot = list(np.zeros(numActions, dtype=np.int32))
    oneHot[action] = 1
    return oneHot




if __name__ == "__main__":
    gameEnv = DoomGameEnv()
    train(gameEnv, doomParams)

