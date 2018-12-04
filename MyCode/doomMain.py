from __future__ import absolute_import

import os
import torch
import numpy as np
import numpy.random as npr
from collections import deque
from time import time

from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution, GameVariable
from DOOM.FrameProcessing import processImage, gameStateToTensor
from DOOM.Params import doomParams
from RAINBOW.ModelMemory import PrioritizedReplayMemory, LinearSchedule, GameState, blank_trans
from RAINBOW.DQN import DQNRecurrent
resultsPath = os.path.join(os.getcwd(), 'Results')

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
    loss = policyNet.f_train(states, actions, returns, isDones, targetNet)  # batchSize*numRecurrentUpdates
    if params.prioritizedReplay:
        loss = (loss.transpose(0,1) * weights).transpose(0,1)  # Multiply by priority weights
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

    # If using prioritized replay, update priorities
    if params.prioritizedReplay:
        memory.updatePriorities(indices, loss.detach().cpu().numpy()[:, -1])

# Evaluate on episodes
def evalEpisode(env, policyNet, numEpisodes=10):
    policyNet.module.eval()
    episodeReturns = np.zeros(numEpisodes)
    episodeLengths = np.zeros(numEpisodes)
    for episode in range(numEpisodes):
        episodeFrameCounter = 0
        isDone = False
        currentState = env.reset()
        while not isDone:
            action = policyNet.next_action(currentState)
            newState, reward, isDone = env.step(oneHotList(action, env.numActions))
            episodeFrameCounter+=1
        episodeReturns[episode] = env.getEpisodeReward()
        episodeLengths[episode] = episodeFrameCounter

    policyNet.module.train()
    return np.mean(episodeReturns), np.mean(episodeLengths)



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
    trainingFrameCounter = 0
    episodeRewards = list()
    evalEpisodeRewards = list()  # For running in eval mode
    episodeLengths = list()
    evalEpisodeLengths = list()  # For running in eval mode
    episodeTimes = list()
    episodeCounter = 1


    while frameCounter < params.numFrames:
        print('Starting episode ' + str(episodeCounter))
        episodeFrameCounter = 0
        isDone = False
        currState = env.reset()
        startTime = time()
        while not isDone:

            # Periodically test in eval mode
            if frameCounter % params.framesBetweenEvaluations == 0:
                print('Evaluating')
                evalReward, evalLength = evalEpisode(env, policyNet)
                evalEpisodeRewards.append(evalReward)
                evalEpisodeLengths.append(evalLength)
                # Start a new episode afterward
                episodeFrameCounter = 0
                isDone = False
                currState = env.reset()
                startTime = time()

            # If using noisy linear, reset noise on training frequency
            if params.noisyLinear and frameCounter % params.trainingFrequency == 0:
                policyNet.resetNoise()

            # Take action based on current state or on eps
            if epsilon is not None and npr.rand() < epsilon.value(trainingFrameCounter):
                action = npr.randint(0, env.numActions)
            else:
                action = policyNet.next_action(currState)

            newState, reward, isDone = env.step(oneHotList(action, env.numActions))
            episodeFrameCounter += 1
            frameCounter += 1
            memory.append(currState[-1], action, reward, isDone)  # Add last frame of game state
            # Populate with random experiences first
            if frameCounter > framesBeforeTraining:
                # If it's time to train
                if frameCounter % params.trainingFrequency == 0:
                    optimizeNet(policyNet, targetNet, memory, optimFunction, params)

                # If it's time to update target net
                if params.double and frameCounter % params.targetUpdateFrequency == 0:
                    updateTargetNet(policyNet, targetNet)

                # Anneal priority beta if needed
                if params.prioritizedReplay:
                    memory.priority_weight = beta.value(trainingFrameCounter)

                trainingFrameCounter += 1

        episodeRewards.append(env.getEpisodeReward())
        episodeLengths.append(episodeFrameCounter)
        episodeTimes.append(time()-startTime)
        episodeCounter+=1

    return episodeRewards, episodeLengths, episodeTimes, evalEpisodeRewards, evalEpisodeLengths, policyNet



def oneHotList(action, numActions):
    oneHot = list(np.zeros(numActions, dtype=np.int32))
    oneHot[action] = 1
    return oneHot

def saveResults(indexStr, rewards, lengths, times, evalRewards, evalLengths, model):
    np.savez(os.path.join(resultsPath, indexStr+'results.npz'), rewards=rewards, lengths=lengths, times=times, evalRewards=evalRewards, evalLengths=evalLengths)
    torch.save(model.module.state_dict(), os.path.join(resultsPath, 'model.pth'))



if __name__ == "__main__":
    bareParams = doomParams(0, 0, 0, 0, 1)
    noPriorityParams = doomParams(0)
    noNoisyParams = doomParams(1, 0)
    noDuelingParams = doomParams(1, 1, 0)
    noDoubleParams = doomParams(1, 1, 1, 0)
    noMultiParams = doomParams(1, 1, 1, 1, 1)
    rainbowParams = doomParams()

    paramList = [bareParams, noPriorityParams, noNoisyParams, noDuelingParams, noDoubleParams, noMultiParams, rainbowParams]

    for index, paramSet in enumerate(paramList):
        gameEnv = DoomGameEnv(paramSet)
        rewards, lengths, times, evalRewards, evalLengths, model = train(gameEnv, paramSet)
        saveResults(str(index), rewards, lengths, times, evalRewards, evalLengths, model)

