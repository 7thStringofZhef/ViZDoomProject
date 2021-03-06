import torch
import numpy as np
import numpy.random as npr
import random
import operator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        try:
            assert 0 <= prefixsum <= self.sum() + 1e-5
        except AssertionError:
            print("Prefix sum error: {}".format(prefixsum))
            exit()
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        return super(MinSegmentTree, self).reduce(start, end)

# Basic memory
class ExperienceReplayMemory:
    def __init__(self, params):
        self.capacity = params.replayMemoryCapacity
        self.memory = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.nextStates = []
        self.seqLen = params.sequenceLength

    def push(self, transition):
        self.states.append(transition[0])
        self.actions.append(transition[1])
        self.rewards.append(transition[2])
        self.nextStates.append(transition[3])
        if len(self.states) > self.capacity:
            del self.states[0], self.actions[0], self.rewards[0], self.nextStates[0]

    def sample(self, batch_size):
        endIndices = npr.randint(self.seqLen, len(self.states), batch_size)
        # Stack frames here if needed
        return [(np.vstack(self.states[idx-self.seqLen:idx]),
                 self.actions[idx],
                 self.rewards[idx],
                 None if any(elem is None for elem in self.nextStates[idx-self.seqLen:idx])
                 else np.vstack(self.nextStates[idx-self.seqLen:idx])) for idx in endIndices], None, None

    def __len__(self):
        return len(self.states)


class PrioritizedReplayMemory(object):
    def __init__(self, params):
        super(PrioritizedReplayMemory, self).__init__()
        self.seqLen = params.sequenceLength
        self._storage = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.nextStates = []
        self._maxsize = params.replayMemoryCapacity
        self._next_idx = 0

        self._alpha = params.priorityOmega

        self.beta_start = params.priorityBetaStart
        self.beta_frames = params.priorityBetaFrames
        self.frame = 1

        it_capacity = 1
        while it_capacity < params.replayMemoryCapacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, data):
        idx = self._next_idx

        if self._next_idx >= len(self._storage):
            self.states.append(data[0])
            self.actions.append(data[1])
            self.rewards.append(data[2])
            self.nextStates.append(data[3])
        else:
            self.states[self._next_idx], self.actions[self._next_idx], self.rewards[self._next_idx], self.nextStates[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _encode_sample(self, idxes):
        # Stack frames here if needed
        return [(np.vstack(self.states[idx-self.seqLen:idx]),
                 self.actions[idx],
                 self.rewards[idx],
                 None if any(elem is None for elem in self.nextStates[idx-self.seqLen:idx])
                 else np.vstack(self.nextStates[idx-self.seqLen:idx])) for idx in idxes]
        # return [self._storage[i-self.seqLen:i] for i in idxes]

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            idx = -1
            while idx < self.seqLen:  # For stacked frames
                mass = npr.random() * self._it_sum.sum(0, len(self._storage) - 1)
                idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size):
        idxes = self._sample_proportional(batch_size)

        weights = []

        # find smallest sampling prob: p_min = smallest priority^alpha / sum of priorities^alpha
        p_min = self._it_min.min() / self._it_sum.sum()

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # max_weight given to smallest prob
        max_weight = (p_min * len(self.states)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.states)) ** (-beta)
            weights.append(weight / max_weight)
        weights = torch.tensor(weights, device=device, dtype=torch.float)
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample, idxes, weights

    def update_priorities(self, idxes, priorities):
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < len(self.states)
            self._it_sum[idx] = (priority + 1e-5) ** self._alpha
            self._it_min[idx] = (priority + 1e-5) ** self._alpha
            self._max_priority = max(self._max_priority, (priority + 1e-5))


class RecurrentExperienceReplayMemory:
    def __init__(self, params):
        self.capacity = params.replayMemoryCapacity
        self.memory = []
        self.seq_length = params.sequenceLength

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        finish = random.sample(range(0, len(self.memory)), batch_size)
        begin = [x - self.seq_length for x in finish]
        samp = []
        for start, end in zip(begin, finish):
            # correct for sampling near beginning
            final = self.memory[max(start + 1, 0):end + 1]

            # correct for sampling across episodes
            for i in range(len(final) - 2, -1, -1):
                if final[i][3] is None:
                    final = final[i + 1:]
                    break

            # pad beginning to account for corrections
            while (len(final) < self.seq_length):
                final = [(np.zeros_like(self.memory[0][0]), 0, 0, np.zeros_like(self.memory[0][3]))] + final

            samp += final

        # returns flattened version
        return samp, None, None

    def __len__(self):
        return len(self.memory)

class RecurrentPrioritizedReplayMemory(object):
    def __init__(self, params):
        super(RecurrentPrioritizedReplayMemory, self).__init__()
        self._storage = []
        self._maxsize = params.replayMemoryCapacity
        self.seqLen = params.sequenceLength
        self._next_idx = 0

        self._alpha = params.priorityOmega

        self.beta_start = params.priorityBetaStart
        self.beta_frames = params.priorityBetaFrames
        self.frame = 1

        it_capacity = 1
        while it_capacity < params.replayMemoryCapacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, data):
        idx = self._next_idx

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _encode_sample(self, idxes):
        return [[self._storage[i] for i in innerIdxes] for innerIdxes in idxes]

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = npr.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    # Only weight the indices of the last frame in each experience
    def sample(self, batch_size):
        idxes = self._sample_proportional(batch_size)
        beginIndices = [idx - self.seqLen for idx in idxes]
        samp = []
        for start, end in zip(beginIndices, idxes):
            # correct for sampling near beginning
            final = self._storage[max(start + 1, 0):end + 1]

            # correct for sampling across episodes
            for i in range(len(final) - 2, -1, -1):
                if final[i][3] is None:
                    final = final[i + 1:]
                    break

            # pad beginning to account for corrections
            while (len(final) < self.seqLen):
                final = [(np.zeros_like(self._storage[0][0]), 0, 0, np.zeros_like(self._storage[0][3]))] + final

            samp += final

        weights = []

        # find smallest sampling prob: p_min = smallest priority^alpha / sum of priorities^alpha
        p_min = self._it_min.min() / self._it_sum.sum()

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # max_weight given to smallest prob
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = torch.tensor(weights, device=device, dtype=torch.float)
        encoded_sample = samp
        return encoded_sample, idxes, weights

    def update_priorities(self, idxes, priorities):
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = (priority + 1e-5) ** self._alpha
            self._it_min[idx] = (priority + 1e-5) ** self._alpha
            self._max_priority = max(self._max_priority, (priority + 1e-5))


