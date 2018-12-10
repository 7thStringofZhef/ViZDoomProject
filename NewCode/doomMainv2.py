import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from copy import deepcopy

from DOOM.Params import doomParams
from DOOM.Environment import VizDoomEnv
from Model.Agent import RainbowAgent


def evalEpisodes(agent, env, numEpisodes=20):
    print('Evaluating on ' +str(numEpisodes) + ' episodes')
    agent.eval()  # Set to eval mode
    evalRewards = 0.0
    state = env.reset()
    for episode in range(numEpisodes):
        action = agent.get_action(np.vstack(state))  # Get action
        newState, reward, isDone, gameVars = env.step(actionList[action])  # Take a step
        evalRewards += reward  # Add reward from action
        state = deepcopy(newState)  # Copy just in case

        # If episode is done, save, reset environment
        if isDone:
            state = env.reset()


    agent.train()  # Set to training mode
    return evalRewards / numEpisodes  # Return mean episode reward

def save(agent, episodeRewards, evalRewards, evalRewardFrames, saveAgent=True):
    print('Saving...')
    agentName = agent.name
    numpyFilename = './saved_agents/'+agentName+'_Results.npz'
    np.savez(numpyFilename, episodeRewards=episodeRewards, evalRewards=evalRewards, evalRewardFrames=evalRewardFrames)
    if saveAgent:
        numpyFilename = './saved_agents/' + agentName + '_AgentResults.npz'
        np.savez(numpyFilename, losses=agent.getLosses(), rewards=agent.getRewards())
        agent.save_w()

def loadAgent(agent):
    agent.load_w()

def loadResults(name):
    return np.load('./saved_agents/'+name+'_Results.npz')

def loadAgentResults(name):
    return np.load('./saved_agents/' + name + '_AgentResults.npz')


if __name__=="__main__":
    bareParams = doomParams("Bare", 0, 0, 0, 0, 1, 0)
    noPriorityParams = doomParams("nPriority", 0)
    noNoisyParams = doomParams("nNoisy", 1, 0)
    noDuelingParams = doomParams("nDueling", 1, 1, 0)
    noDoubleParams = doomParams("nDouble", 1, 1, 1, 0)
    noMultiParams = doomParams("nMulti", 1, 1, 1, 1, 1)
    noDistributedParams = doomParams("nDistributed", 1, 1, 1, 1, 3, 0)
    rainbowParams = doomParams("Rainbow")
    paramList = [bareParams, rainbowParams, noPriorityParams, noNoisyParams, noDuelingParams, noDoubleParams,
                 noMultiParams, noDistributedParams]

    #One-hot for the environment
    actionList = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]

    for params in paramList:
        try:
            torch.cuda.empty_cache()
            print(params.modelName)
            env = VizDoomEnv(params)
            agent = RainbowAgent(params)
            currentFrame = 0
            episodeRewards = list()
            evalRewards = list()
            evalRewardFrames = list()
            currentEpisodeReward = 0.0
            state = env.reset()
            episodeCounter = 1
            nextEvalFrame = params.framesBetweenEvaluations
            nextSaveFrame = params.framesBetweenSaves

            while currentFrame <= params.numFrames:
                currentFrame += 1
                if currentFrame == params.framesBeforeTraining:
                    print("Training starting...")

                # Evaluate periodically (takes noisy layer offline)
                if currentFrame >= nextEvalFrame:
                    evalRewardFrames.append(currentFrame)
                    evalRewards.append(evalEpisodes(agent, env))
                    nextEvalFrame += params.framesBetweenEvaluations
                    state = env.reset()

                # Save periodically (in case stuff dies)
                if currentFrame >= nextSaveFrame:
                    save(agent, episodeRewards, evalRewards, evalRewardFrames)
                    nextSaveFrame += params.framesBetweenSaves

                action = agent.get_action(np.vstack(state))  # Get action
                agent.currFrame += 1  # Update agent's counter
                newState, reward, isDone, gameVars = env.step(actionList[action])  # Take a step
                if isDone:
                    newState = [None]
                agent.update(state[-1], action, reward, newState[-1], currentFrame)  # Add experience to memory, train
                currentEpisodeReward += reward  # Add reward from action
                state = deepcopy(newState)  # Copy just in case

                # If episode is done, save, reset environment
                if isDone:
                    episodeCounter += 1
                    print('Starting episode ' + str(episodeCounter))
                    agent.finish_nstep()
                    state = env.reset()
                    agent.save_reward(currentEpisodeReward)
                    episodeRewards.append(currentEpisodeReward)
                    currentEpisodeReward = 0.0

            # Final save
            save(agent, episodeRewards, evalRewards, evalRewardFrames)
            env.game.close()
            # agent.save_replay()
        except Exception as e:
            print(e)
            continue


"""
    for params in paramList:
        env = VizDoomEnv(params)
        currentFrame=0

        # Run for max # frames
        while currentFrame < params.numFrames:
            pass
            """
