import numpy.random as npr
from collections import namedtuple
import torch
import numpy as np

Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))
blank_trans = Transition(0, torch.zeros(108, 60, 3, dtype=torch.uint8), None, 0, False)

# For the beta value in the distribution
class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_B=1, initial_B=0.4):
        '''
        Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Args:
            - schedule_timesteps: Number of timesteps for which to linearly anneal initial_p to final_p
            - initial_p: initial output value
            -final_p: final output value
        '''
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_B
        self.initial_p = initial_B

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

class ReplayMemory(object):
    def __init__(self, capacity=200000):
        self.capacity = capacity
        self.index = 0
        self.memory = []

    def add(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(*args)
        self.index = (self.index+1) % self.capacity  # Circular buffer

    def sample(self, batchSize, historySize=1):
        currSize = len(self.memory)
        if currSize == self.capacity:
            indices = npr.choice(currSize, batchSize)
        else:
            indices = npr.choice(currSize-(historySize-1), batchSize)
        endIndices = indices + (historySize-1)
        transitions = [self.memory[index:endIndex] for index, endIndex in zip(indices, endIndices)]
        return transitions, indices

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):
    def __init__(self, capacity=200000, priorityOmega=0.5, priorityBetaSchedule=LinearSchedule(200000, 1, 0.4)):
        self.capacity = capacity
        self.index = 0
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.betaSchedule = priorityBetaSchedule
        self.omega = priorityOmega

    def add(self, *args):
        maxPriority = self.priorities.max() if self.buffer else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(*args)
        self.priorities[self.index] = maxPriority  # Starts at max priority as newest sample
        self.index = (self.index+1) % self.capacity  # Circular buffer

    def sample(self, batchSize, currentTimestep, historySize=1):
        probs = self.priorities[:self.index] ** self.omega
        probs = probs / probs.sum()
        currSize = len(self.memory)
        indices = npr.choice(currSize, batchSize, p=probs)
        transitions = [self.memory[index] for index in indices]
        weights = (currSize * probs[indices]) ** (-self.betaSchedule.value(currentTimestep))
        weights = weights / weights.max()
        return transitions, weights, indices

    def updatePriorities(self, batchIndices, batchPriorities):
        self.priorities[batchIndices.astype(np.int32)] = batchPriorities

    def __len__(self):
        return len(self.memory)