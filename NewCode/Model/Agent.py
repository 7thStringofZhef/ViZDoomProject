import numpy as np
import pickle
import os.path

import torch
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .Feedforward import *
from .Recurrent import *
from .Memory import ExperienceReplayMemory, PrioritizedReplayMemory

# Map params to specific model
def chooseModel(params):
    if params.dueling:
        if params.recurrent:
            if params.distributed:
                return CategoricalDuelingDRQN
            else:
                return DuelingDRQN
        else:
            if params.distributed:
                return CategoricalDuelingDQN
            else:
                return DuelingDQN
    else:
        if params.recurrent:
            if params.distributed:
                return CategoricalDRQN
            else:
                return DRQN
        else:
            if params.distributed:
                return CategoricalDQN
            else:
                return DQN

class LinearSchedule(object):
    def __init__(self, startTimesteps, schedule_timesteps, final_B=1, initial_B=0.4):
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
        self.trainingStartTimesteps = float(startTimesteps)
        self.final_p = final_B
        self.initial_p = initial_B

    def value(self, t):
        fraction = min((max(self.trainingStartTimesteps-float(t), 0)) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class BaseAgent(object):
    def __init__(self):
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.losses = []
        self.rewards = []
        self.sigma_parameter_mag = []

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def save_w(self):
        torch.save(self.model.state_dict(), './saved_agents/model.dump')
        torch.save(self.optimizer.state_dict(), './saved_agents/optim.dump')

    def load_w(self):
        fname_model = "./saved_agents/model.dump"
        fname_optim = "./saved_agents/optim.dump"

        if os.path.isfile(fname_model):
            self.model.load_state_dict(torch.load(fname_model))
            self.target_model.load_state_dict(self.model.state_dict())

        if os.path.isfile(fname_optim):
            self.optimizer.load_state_dict(torch.load(fname_optim))

    def save_replay(self):
        pickle.dump(self.memory, open('./saved_agents/exp_replay_agent.dump', 'wb'))

    def load_replay(self):
        fname = './saved_agents/exp_replay_agent.dump'
        if os.path.isfile(fname):
            self.memory = pickle.load(open(fname, 'rb'))

    def save_sigma_param_magnitudes(self):
        tmp = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'sigma' in name:
                    tmp += param.data.cpu().numpy().ravel().tolist()
        if tmp:
            self.sigma_parameter_mag.append(np.mean(np.abs(np.array(tmp))))

    def save_loss(self, loss):
        self.losses.append(loss)

    def save_reward(self, reward):
        self.rewards.append(reward)


class RainbowAgent(BaseAgent):
    def __init__(self, params, env=None):
        super(RainbowAgent, self).__init__()
        self.params = params
        self.device = device

        self.noisy = params.noisyLinear
        self.priorityReplay = params.prioritizedReplay

        # Model parameters
        self.gamma = params.gamma
        self.lr = params.lr
        self.targetUpdateFrequency = params.targetUpdateFrequency
        self.replayCapacity = params.replayMemoryCapacity
        self.batchSize = params.batchSize
        self.learn_start = params.framesBeforeTraining
        self.sigma_init = params.noisyParam
        self.priority_beta_start = params.priorityBetaStart
        self.priority_beta_end = params.priorityBetaEnd
        self.priority_beta_frames = params.priorityBetaFrames
        self.priority_alpha = params.priorityOmega
        self.atoms = params.atoms
        self.vMin = params.vMin
        self.vMax = params.vMax
        self.supports = torch.linspace(self.vMin, self.vMax, self.atoms).view(1, 1, self.atoms).to(device)
        self.delta = (self.vMax - self.vMin) / (self.atoms - 1)

        self.num_feats = params.inputShape
        self.num_actions = params.numActions
        self.env = env

        self.declare_networks()  # Set up models depending on parameters

        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        # Set to training mode
        self.model.train()
        self.target_model.train()

        self.update_count = 0

        # Create replay memory
        self.declare_memory()

        self.nsteps = params.multiStep
        self.nstep_buffer = []

        # Linear schedules to anneal params over time
        self.epsSchedule = LinearSchedule(params.framesBeforeTraining, params.epsSteps)

        # Current timestep
        self.currFrame = 0

    def declare_networks(self):
        modelFn = chooseModel(self.params)
        self.model = modelFn(self.params)
        if self.params.double:
            self.target_model = modelFn(self.params)
        else:
            self.target_model = None

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.params) if not self.priorityReplay \
            else PrioritizedReplayMemory(self.params)

    def append_to_replay(self, s, a, r, s_):
        self.nstep_buffer.append((s, a, r, s_))

        if (len(self.nstep_buffer) < self.nsteps):
            return

        R = sum([self.nstep_buffer[i][2] * (self.gamma ** i) for i in range(self.nsteps)])
        state, action, _, _ = self.nstep_buffer.pop(0)

        self.memory.push((state, action, R, s_))

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batchSize)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,) + self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device,
                                      dtype=torch.uint8)
        try:  # sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device,
                                                 dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        # estimate
        self.model.sample_noise()
        current_q_values = self.model(batch_state).gather(1, batch_action)

        # target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batchSize, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                self.target_model.sample_noise()
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + ((self.gamma ** self.nsteps) * max_next_q_values)

        diff = (expected_q_values - current_q_values)
        if self.priorityReplay:
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = self.huber(diff).squeeze() * weights
        else:
            loss = self.huber(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, frame=0):
        if self.static_policy:
            return None

        self.append_to_replay(s, a, r, s_)

        if frame < self.learn_start:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        self.save_loss(loss.item())
        self.save_sigma_param_magnitudes()

    def get_action(self, s):
        eps = self.epsSchedule.value(self.currFrame)
        with torch.no_grad():
            if np.random.random() >= eps or self.noisy:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                self.model.sample_noise()
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.targetUpdateFrequency
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2] * (self.gamma ** i) for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.push((state, action, R, None))

    def reset_hx(self):
        pass

    def eval(self):
        self.evalMode = 1
        self.model.eval()
        if self.target_model is not None:
            self.target_model.eval()

    def train(self):
        self.evalMode = 0
        self.model.train()
        if self.target_model is not None:
            self.target_model.train()
