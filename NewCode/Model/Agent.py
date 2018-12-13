import numpy as np
import pickle
import os.path

import torch
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .Feedforward import *
from .Recurrent import *
from .Memory import ExperienceReplayMemory, PrioritizedReplayMemory, RecurrentExperienceReplayMemory, RecurrentPrioritizedReplayMemory

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
        self.name = "Temp"

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def save_w(self):
        torch.save(self.model.state_dict(), './saved_agents/'+self.name+'model.dump')
        torch.save(self.optimizer.state_dict(), './saved_agents/'+self.name+'optim.dump')

    def load_w(self):
        fname_model = "./saved_agents/"+self.name+"model.dump"
        fname_optim = "./saved_agents/"+self.name+"optim.dump"

        if os.path.isfile(fname_model):
            self.model.load_state_dict(torch.load(fname_model))
            self.target_model.load_state_dict(self.model.state_dict())

        if os.path.isfile(fname_optim):
            self.optimizer.load_state_dict(torch.load(fname_optim))

    def save_replay(self):
        pickle.dump(self.memory, open('./saved_agents/'+self.name+'exp_replay_agent.dump', 'wb'))

    def load_replay(self):
        fname = './saved_agents/'+self.name+'exp_replay_agent.dump'
        if os.path.isfile(fname):
            self.memory = pickle.load(open(fname, 'rb'))

    def save_sigma_param_magnitudes(self):
        tmp = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'Sigma' in name:
                    tmp += param.data.cpu().numpy().ravel().tolist()
        if tmp:
            self.sigma_parameter_mag.append(np.mean(np.abs(np.array(tmp))))

    def save_loss(self, loss):
        self.losses.append(loss)

    def getLosses(self):
        return self.losses

    def save_reward(self, reward):
        self.rewards.append(reward)

    def getRewards(self):
        return self.rewards


class RainbowAgent(BaseAgent):
    def __init__(self, params, env=None):
        super(RainbowAgent, self).__init__()
        self.params = params
        self.name = params.modelName
        self.device = device

        self.noisy = params.noisyLinear
        self.priorityReplay = params.prioritizedReplay

        # Model parameters
        self.gamma = params.gamma
        self.lr = params.learningRate
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

        if params.recurrent:  # Feed frames one at a time
            self.num_feats = params.inputShape
        else:  # Stack frames
            self.num_feats = (params.inputShape[0]*params.sequenceLength, params.inputShape[1], params.inputShape[2])
        self.num_actions = params.numActions
        self.env = env

        self.declare_networks()  # Set up models depending on parameters

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

        # To calculate n-step returns
        self.nsteps = params.multiStep
        self.nstep_buffer = []

        # Linear schedules to anneal params over time
        self.epsSchedule = LinearSchedule(params.framesBeforeTraining, params.epsSteps)

        # Current timestep
        self.currFrame = 0

        # Sequence length
        self.sequence_length = params.sequenceLength

        if params.recurrent:
            self.reset_hx()

    def declare_networks(self):
        modelFn = chooseModel(self.params)
        self.model = modelFn(self.params)
        self.target_model = modelFn(self.params)
        self.target_model.load_state_dict(self.model.state_dict())

    def declare_memory(self):
        if not self.params.recurrent:
            self.memory = ExperienceReplayMemory(self.params) if not self.priorityReplay \
                else PrioritizedReplayMemory(self.params)
        else:
            self.memory = RecurrentExperienceReplayMemory(self.params) if not self.priorityReplay \
                else RecurrentPrioritizedReplayMemory(self.params)

    def append_to_replay(self, s, a, r, s_):
        self.nstep_buffer.append((s, a, r, s_))

        if (len(self.nstep_buffer) < self.nsteps):
            return

        R = sum([self.nstep_buffer[i][2] * (self.gamma ** i) for i in range(self.nsteps)])
        state, action, _, _ = self.nstep_buffer.pop(0)

        self.memory.push((state, action, R, s_))

    # Prepare a batch from memory for training
    def prep_minibatch(self):
        if self.params.recurrent:
            return self.prep_minibatch_recurrent()
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

    def prep_minibatch_recurrent(self):
        transitions, indices, weights = self.memory.sample(self.batchSize)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (self.batchSize, self.sequence_length) + self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).view(self.batchSize,
                                                                                             self.sequence_length, -1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).view(self.batchSize,
                                                                                              self.sequence_length)
        # get set of next states for end of each sequence
        batch_next_state = tuple(
            [batch_next_state[i] for i in range(len(batch_next_state)) if (i + 1) % (self.sequence_length) == 0])

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device,
                                      dtype=torch.uint8)
        try:  # sometimes all next states are false, especially with nstep returns
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device,
                                                 dtype=torch.float).unsqueeze(dim=1)
            non_final_next_states = torch.cat([batch_state[non_final_mask, 1:, :], non_final_next_states], dim=1)
            empty_next_state_values = False
        except:
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    # For distributed. Calculate distribution across Q-values
    def projection_distribution(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        with torch.no_grad():
            max_next_dist = torch.zeros((self.batchSize, 1, self.atoms), device=self.device,
                                        dtype=torch.float) + 1. / self.atoms
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action_dist(non_final_next_states)  # Best action next state
                self.target_model.sample_noise()
                max_next_dist[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)  # From target model, dist values
                max_next_dist = max_next_dist.squeeze()

            Tz = batch_reward.view(-1, 1) + (self.gamma ** self.nsteps) * self.supports.view(1, -1) * non_final_mask.to(
                torch.float).view(-1, 1)
            Tz = Tz.clamp(self.vMin, self.vMax)
            b = (Tz - self.vMin) / self.delta
            l = b.floor().to(torch.int64)
            u = b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            offset = torch.linspace(0, (self.batchSize - 1) * self.atoms, self.batchSize).unsqueeze(dim=1).expand(
                self.batchSize, self.atoms).to(batch_action)
            m = batch_state.new_zeros(self.batchSize, self.atoms)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (max_next_dist * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (max_next_dist * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        return m

    def compute_loss(self, batch_vars):
        if self.params.recurrent:
            if self.params.distributed:
                return self.compute_loss_recurrent_dist(batch_vars)
            else:
                return self.compute_loss_recurrent(batch_vars)
        else:
            if self.params.distributed:
                return self.compute_loss_dist(batch_vars)
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

    def compute_loss_dist(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        batch_action = batch_action.unsqueeze(dim=-1).expand(-1, -1, self.atoms)
        batch_reward = batch_reward.view(-1, 1, 1)

        # estimate
        self.model.sample_noise()
        current_dist = self.model(batch_state).gather(1, batch_action).squeeze()

        target_prob = self.projection_distribution(batch_vars)

        loss = -(target_prob * current_dist.log()).sum(-1)
        if self.priorityReplay:
            self.memory.update_priorities(indices, loss.detach().squeeze().abs().cpu().numpy().tolist())
            loss = loss * weights
        loss = loss.mean()

        return loss

    def compute_loss_recurrent(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        # estimate
        current_q_values, _ = self.model(batch_state)
        current_q_values = current_q_values.gather(2, batch_action).squeeze()

        # target
        with torch.no_grad():
            max_next_q_values = torch.zeros((self.batchSize, self.sequence_length), device=self.device,
                                            dtype=torch.float)
            if not empty_next_state_values:
                max_next, _ = self.target_model(non_final_next_states)
                max_next_q_values[non_final_mask] = max_next.max(dim=2)[0]
            expected_q_values = batch_reward + ((self.gamma ** self.nsteps) * max_next_q_values)

        diff = (expected_q_values - current_q_values)
        loss = self.huber(diff)
        loss = loss.mean()
        return loss

    #***Still needs work
    def compute_loss_recurrent_dist(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        # estimate
        current_q_values, _ = self.model(batch_state)
        current_q_values = current_q_values.gather(2, batch_action).squeeze()

        # target
        with torch.no_grad():
            max_next_q_values = torch.zeros((self.batchSize, self.sequence_length), device=self.device,
                                            dtype=torch.float)
            if not empty_next_state_values:
                max_next, _ = self.target_model(non_final_next_states)
                max_next_q_values[non_final_mask] = max_next.max(dim=2)[0]
            expected_q_values = batch_reward + ((self.gamma ** self.nsteps) * max_next_q_values)

        diff = (expected_q_values - current_q_values)
        loss = self.huber(diff)
        loss = loss.mean()
        return loss

    # Append experience to replay buffer
    def update(self, s, a, r, s_, frame=0):
        self.append_to_replay(s, a, r, s_)

        # If it's not time to start training, don't
        if frame < self.learn_start:
            return None

        # Train if it's time to train
        if frame % self.params.trainingFrequency == 0:
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

    # Following epsGreedy policy (or not with noisy linear), get action
    def get_action(self, s):
        if self.params.distributed:
            return self.get_action_dist(s)
        eps = self.epsSchedule.value(self.currFrame)
        with torch.no_grad():
            if np.random.random() >= eps or self.noisy:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                self.model.sample_noise()
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    # As above, but for distribution
    def get_action_dist(self, s):
        eps = self.epsSchedule.value(self.currFrame)
        with torch.no_grad():
            if np.random.random() >= eps or self.noisy:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                self.model.sample_noise()
                a = self.model(X) * self.supports
                a = a.sum(dim=2).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)


    # Update target model according to frequency
    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.targetUpdateFrequency
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    # Get best action (standard)
    def get_max_next_state_action(self, next_states):
        if not self.params.double:
            return self.target_model(next_states).max(dim=1)[1].view(-1, 1)
        else:
            return self.model(next_states).max(dim=1)[1].view(-1, 1)


    # Get best action (distributed)
    def get_max_next_state_action_dist(self, next_states):
        if self.params.double:
            next_dist = self.model(next_states) * self.supports
        else:
            next_dist = self.target_model(next_states) * self.supports
        return next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self.atoms)

    # At end of episode, get the last n step return and append to memory
    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2] * (self.gamma ** i) for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.push((state, action, R, None))

    # Reset state buffer
    def reset_hx(self):
        self.seq = [np.zeros(self.num_feats) for j in range(self.sequence_length)]

    # Set to eval
    def eval(self):
        self.evalMode = 1
        self.model.eval()
        self.target_model.eval()

    # Set to train
    def train(self):
        self.evalMode = 0
        self.model.train()
        self.target_model.train()
