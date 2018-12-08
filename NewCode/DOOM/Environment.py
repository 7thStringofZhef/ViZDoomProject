import numpy as np
import gym
import gym.spaces as spaces
from gym.envs.classic_control import rendering
from vizdoom import DoomGame
import cv2

class VizDoomEnv(gym.Env):
    '''
    Wrapper for vizdoom to use as an OpenAI gym environment.
    '''
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, params):
        super(VizDoomEnv, self).__init__()
        self.params = params
        self.game = DoomGame()
        self.game.load_config(params.scenarioPath)
        self._viewer = None
        self.frameskip = params.frameskip
        self.inputShape = params.inputShape
        self.gameVariables = params.gameVariables
        self.numGameVariables = len(self.gameVariables)
        self.action_space = spaces.MultiDiscrete([2] * self.game.get_available_buttons_size())
        self.action_space.dtype = 'uint8'
        output_shape = (self.game.get_screen_channels(), self.game.get_screen_height(), self.game.get_screen_width())
        self.observation_space = spaces.Box(low=0, high=255, shape=output_shape, dtype='uint8')
        self.game.init()

    def close(self):
        self.game.close()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def seed(self, seed=None):
        self.game.set_seed(seed)

    def step(self, action):
        reward = self.game.make_action(list(action), self.frameskip)
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        if state is not None:
            observation = state.screen_buffer
            info = state.game_variables  # Return the chosen game variables in info
        else:
            observation = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)
            info = None
        processedObservation = self._preProcessImage(observation)
        return processedObservation, reward, done, info

    # Preprocess image for use in network
    def _preProcessImage(self, image):
        if image.shape != self.inputShape:
            image = cv2.resize(
                image.transpose(1, 2, 0),
                (self.inputShape[2], self.inputShape[1]),
                interpolation=cv2.INTER_AREA
            ).transpose(2, 0, 1)
        return image

    def reset(self):
        self.game.new_episode()
        return self._preProcessImage(self.game.get_state().screen_buffer)

    def render(self, mode='human', close=False):
        if close:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            return
        img = None
        state = self.game.get_state()
        if state is not None:
            img = state.screen_buffer
        if img is None:
            # at the end of the episode
            img = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)
        if mode == 'rgb_array':
            return img
        elif mode is 'human':
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer()
            self._viewer.imshow(img.transpose(1, 2, 0))