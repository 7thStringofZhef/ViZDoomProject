import numpy.random as npr
from collections import namedtuple
import torch
import numpy as np

Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'isDone'))
GameState = namedtuple('State', ('buffer', 'gameVars'))
blank_trans = Transition(0, GameState(torch.zeros(3, 60, 108, dtype=torch.uint8),
                                      torch.zeros(2, dtype=torch.float32)), None, 0, True)

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


# Store priorities as binary tree
class SegmentTree(object):
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.sum_tree = [0] * (2 * size - 1)  # Initialise fixed size tree with all (priority) zeros
        self.data = [None] * size
        self.max = 1  # Initial max value to return

  # Propagates value up tree given a tree index
    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

  # Updates value given a tree index
    def update(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate(index, value)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self.update(self.index + self.size - 1, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

  # Searches for the location of a value in sum tree
    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])

  # Searches for a value in sum tree and returns value, data index and tree index
    def find(self, value):
        index = self._retrieve(0, value)  # Search for index of item from root
        data_index = index - self.size + 1
        return (self.sum_tree[index], data_index, index)  # Return value, data index, tree index

      # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]



class PrioritizedReplayMemory(object):
    def __init__(self, params):
        self.capacity = params.replayMemoryCapacity
        self.history = params.recurrenceHistory + params.numRecurrentUpdates
        self.gamma = params.gamma
        self.multiStepN = params.multiStep
        self.priority_weight = params.priorityBetaStart # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = params.priorityOmega
        self.t = 0  # Internal episode timestep counter
        self.transitions = SegmentTree(params.replayMemoryCapacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

  # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, state, action, reward, isDone):
        self.transitions.append(Transition(self.t, state, action, reward, isDone), self.transitions.max)  # Store new transition with maximum priority
        self.t = 0 if isDone else self.t + 1  # Start new episodes with t = 0

  # Returns a transition with blank states where appropriate
    def _get_transition(self, idx):
        transition = [None] * (self.history + self.multiStepN)
        transition[self.history - 1] = self.transitions.get(idx) # Last frame before multi-step returns
        for t in range(self.history - 2, -1, -1):  # iterate back to fill in history
            if transition[t + 1].timestep == 0:
                transition[t] = blank_trans  # If we're at beginning, give it a blank
            else:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)
        for t in range(self.history, self.history + self.multiStepN):  # Multi-step return frames. If n=1, then it's just one frame and required history
            if not transition[t - 1].isDone:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)
            else:
                transition[t] = blank_trans  # If prev (next) frame is terminal
        return transition

  # Returns a valid sample from a segment
  # Need to sample one at a time to make sure we pick something with enough frames around it
    def _get_sample_from_segment(self, segment, i):
        valid = False
        while not valid:
            sample = np.random.uniform(i * segment, (i + 1) * segment)  # Uniformly sample an element from within a segment
            prob, idx, tree_idx = self.transitions.find(sample)  # Retrieve sample from tree with un-normalised probability
            # Resample if transition straddled current index or probablity 0
            if (self.transitions.index - idx) % self.capacity > self.multiStepN and (idx - self.transitions.index) % self.capacity >= self.history and prob != 0:
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        # Retrieve all required transition data (from t - h to t + n)
        transition = self._get_transition(idx)

        #Retrun GameState objects for each state
        state = [trans.state for trans in transition[:self.history]]
        next_state = [trans.state for trans in transition[self.multiStepN:self.multiStepN + self.history]]

        # Discrete action to be used as index
        action = torch.LongTensor([transition[self.history - 1].action]).cuda()

        # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        R = torch.FloatTensor([sum(self.gamma ** n * transition[self.history + n - 1].reward for n in range(self.multiStepN))]).cuda()

        # Mask for non-terminal nth next states
        isDone = torch.FloatTensor([transition[self.history + self.multiStepN - 1].isDone]).cuda()

        return prob, idx, tree_idx, state, action, R, next_state, isDone

    def sample(self, batch_size):
        p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
        segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]  # Get batch of valid samples
        probs, idxs, tree_idxs, states, actions, returns, next_states, isDones = zip(*batch)
        #states, next_states, = torch.stack(states), torch.stack(next_states)
        #actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(isDones)
        probs = np.array(probs, dtype=np.float32) / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = torch.FloatTensor(weights / weights.max()).cuda()  # Normalise by max importance-sampling weight from batch
        return tree_idxs, states, actions, returns, next_states, isDones, weights


    def updatePriorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

  # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

  # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
          raise StopIteration
        # Create stack of states
        state_stack = [None] * self.history
        state_stack[-1] = self.transitions.data[self.current_idx].state
        prev_timestep = self.transitions.data[self.current_idx].timestep
        for t in reversed(range(self.history - 1)):
          if prev_timestep == 0:
            state_stack[t] = blank_trans.state  # If future frame has timestep 0
          else:
            state_stack[t] = self.transitions.data[self.current_idx + t - self.history + 1].state
            prev_timestep -= 1
        state = torch.stack(state_stack, 0).to(dtype=torch.float32, device='cuda').div_(255)  # Agent will turn into batch
        self.current_idx += 1
        return state