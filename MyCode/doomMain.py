from __future__ import absolute_import

import configparser
import torch
from functools import partial
import numpy as np

from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution, GameVariable
from DOOM.FrameProcessing import processImage
from RAINBOW.ModelMemory import PrioritizedReplayMemory, LinearSchedule

def parseHyperparamIni(filepath='Hyperparams.ini'):
    netConfig = configparser.ConfigParser()
    netConfig.read(filepath)
    rainbowUse = netConfig['RAINBOW COMPONENTS']
    rainbowParams = netConfig['RAINBOW PARAMS']
    netParams = netConfig['NET PARAMS']
    return rainbowUse, netParams, rainbowParams

def initGameWithParams(configFilePath):
    game = DoomGame()
    game.load_config(configFilePath)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_400X225)
    game.init()
    return game

def runEpisode(game):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()


class DoomGameEnv(object):
    def __init__(self, scenarioPath='CS7180Scenario.cfg', hyperparamPath='Hyperparams.ini'):
        self.rainbowUse, self.netParams, self.rainbowParams = parseHyperparamIni(hyperparamPath)
        self.frameskip = self.netParams['frameskip']
        self.historySize = self.netParams['recurrenceHistory']
        self.game = initGameWithParams(scenarioPath)
        self.imageHeight, self.imageWidth = (60, 108)
        self.preprocessImage = partial(processImage, newHeight=self.imageHeight, newWidth=self.imageWidth)
        self.actions = game.get_available_buttons()
        self.numActions = len(self.actions)
        self.actions = np.identity(self.numActions, dtype=np.int32).tolist()  # One-hot encoding of actions

    def step(self, action):
        reward = 0
        is_done = False
        self.game.set_action(action)
        for _ in range(self.frameskip):
            reward += game.advance_action()
        is_done = self.game.is_episode_finished() or self.game.is_player_dead()
        newState = self.game.get_state()
        return newState, reward, is_done







if __name__ == "__main__":
    game = initGameWithParams("CS7180Scenario.cfg")
    rainbowUse, netParams, rainbowParams = parseHyperparamIni()
    numActions = game.get_available_buttons_size()
    imageHeight, imageWidth = (60, 108)

    # Define basic network

    #betaSchedule = LinearSchedule(200000, 1, 0.4)
    #epsSchedule = LinearSchedule(100000, 0.1, 1)  # If using, eps decay

    pass

