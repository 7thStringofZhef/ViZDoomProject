import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from DOOM.Params import doomParams
from DOOM.Environment import VizDoomEnv
from Model.Agent import RainbowAgent

if __name__=="__main__":
    bareParams = doomParams(0, 0, 0, 0, 1, 0)
    noPriorityParams = doomParams(0)
    noNoisyParams = doomParams(1, 0)
    noDuelingParams = doomParams(1, 1, 0)
    noDoubleParams = doomParams(1, 1, 1, 0)
    noMultiParams = doomParams(1, 1, 1, 1, 1)
    noDistributedParams = doomParams(1, 1, 1, 1, 3, 0)
    rainbowParams = doomParams()
    paramList = [bareParams, rainbowParams, noPriorityParams, noNoisyParams, noDuelingParams, noDoubleParams,
                 noMultiParams, noDistributedParams]

    for params in paramList:
        testAgent = RainbowAgent(params)

"""
    for params in paramList:
        env = VizDoomEnv(params)
        currentFrame=0

        # Run for max # frames
        while currentFrame < params.numFrames:
            pass
            """
