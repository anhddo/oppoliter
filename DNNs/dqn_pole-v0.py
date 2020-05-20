import gym
import math
import random
import timeit
import numpy as np
from collections import namedtuple
from itertools import count
import argparse
from os import path
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm, trange
from  torch.optim import lr_scheduler 

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# default `log_dir` is "runs" - we'll be more specific here

parser = argparse.ArgumentParser(description="Finite-horizon MDP")
parser.add_argument("--env", default='CartPole-v0')
parser.add_argument("--step", type=int, default=10000)
parser.add_argument("--embeded-size", type=int, default=16)
parser.add_argument("--target-update", type=int, default=500)
parser.add_argument("--optimistic", action="store_true")
parser.add_argument("--algo", default='')
parser.add_argument("--render", action="store_true")
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--beta", type=float, default=0.01)
parser.add_argument("--start-index", type=int, default=0)
parser.add_argument("--saved-dir")
args = parser.parse_args()
setting = vars(args)

writer = SummaryWriter('logs/{}-{}'.format(setting['algo'], datetime.now()))
env = gym.make(setting['env'])
n_actions = env.action_space.n
n_observations = env.observation_space.shape[0]

torch.set_default_tensor_type(torch.DoubleTensor)

# set up matplotlib


device = torch.device("cuda" if torch.cuda.is_available() and not setting['cpu'] else "cpu")



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
BATCH_SIZE = setting['batch_size']
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = setting['step'] / 10



class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Memory:
    def __init__(self, max_len, state_shape, state_dtype):
        assert isinstance(max_len, int)
        assert max_len > 0

        self.max_len = max_len                      # maximum length        
        self._curr_insert_ptr = 0                   # index to insert next data sample
        self._curr_len = 0                          # number of currently stored elements

        state_arr_shape = [max_len] + list(state_shape)

        self._hist_St = np.zeros(state_arr_shape, dtype=state_dtype)
        self._hist_At = np.zeros(max_len, dtype=int)
        self._hist_Rt_1 = np.zeros(max_len, dtype=float)
        self._hist_St_1 = np.zeros(state_arr_shape, dtype=state_dtype)
        self._hist_done_1 = np.zeros(max_len, dtype=bool)

    def append(self, St, At, Rt_1, St_1, done_1):
        self._hist_St[self._curr_insert_ptr] = St
        self._hist_At[self._curr_insert_ptr] = At
        self._hist_Rt_1[self._curr_insert_ptr] = Rt_1
        self._hist_St_1[self._curr_insert_ptr] = St_1
        self._hist_done_1[self._curr_insert_ptr] = done_1
        if self._curr_len < self.max_len:                 # keep track of current length
            self._curr_len += 1
        self._curr_insert_ptr += 1                        # increment insertion pointer
        if self._curr_insert_ptr >= self.max_len:         # roll to zero if needed
            self._curr_insert_ptr = 0

    def __len__(self):
        return self._curr_len

    def get_batch(self, batch_len):
        assert self._curr_len > 0
        assert batch_len > 0

        indices = np.random.randint(        # randint much faster than np.random.sample
            low=0, high=self._curr_len, size=batch_len, dtype=int)

        states = np.take(self._hist_St, indices, axis=0)
        actions = np.take(self._hist_At, indices, axis=0)
        rewards_1 = np.take(self._hist_Rt_1, indices, axis=0)
        states_1 = np.take(self._hist_St_1, indices, axis=0)
        dones_1 = np.take(self._hist_done_1, indices, axis=0)

        return states, actions, rewards_1, states_1, dones_1, indices




class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        n = 24
        self.ln1 = nn.Linear(input_dim, n)
        self.ln2 = nn.Linear(n, n)
        #self.ln3 = nn.Linear(n, n)
        #self.ln4 = nn.Linear(n, n)
        self.ln5 = nn.Linear(n, setting['embeded_size'])
        #self.conv1 = nn.Linear(input_dim, 512)
        #self.conv2 = nn.Linear(512, 256)
        #self.conv3 = nn.Linear(256, EMBEDED_SIZE)
        self.head = nn.Linear(setting['embeded_size'], output_dim)
        self.cov = [torch.eye(setting['embeded_size']).to(device) * 1e-5] * n_actions
        self.inv_cov = [torch.inverse(M) for M in self.cov]

    def embeded_vector(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        #x = F.relu(self.ln3(x))
        #x = F.relu(self.ln4(x))
        x = F.relu(self.ln5(x))
        return x

    def q(self, x):
        v = self.embeded_vector(x)
        o = self.head(v)
        return o, v

    def bonus(self, v):
        b = [setting['beta'] * torch.sqrt(v.mm(M).mm(v.T).diagonal()) for M in self.inv_cov]
        b = torch.stack(b, axis=1)
        return b

    def forward(self, x):
        o, v = self.q(x)
        if setting['algo'] == 'optimistic':
            with torch.no_grad():
                b = self.bonus(v)
            #b = b.detach()
                o += b
        o = torch.clamp(o, max=env._max_episode_steps)
        return o







steps_done = 0


policy_net, target_net = None, None

def no_explore(policy_net, state):
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        action = policy_net(state).max(1)[1].view(1,1)
        assert action <=n_actions
        return action

eps_threshold=1.
def epsilon_greedy(policy_net, state, t):
    global eps_threshold
    sample = random.random()
    eps_threshold = max(EPS_END, eps_threshold * 0.999)
    #eps_threshold = EPS_END + (EPS_START - EPS_END) *  math.exp(- t / (setting['step'] / 10))
    #state = torch.tensor(state).view(-1, n_observations)
    writer.add_scalar('dqn/epsilon', eps_threshold, t)
    if sample > eps_threshold:
        return no_explore(policy_net, state)
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        assert action <=n_actions
        return action


episode_durations = []



def optimize_model(policy_net, target_net, optimizer, memory, t):
    if len(memory) < BATCH_SIZE:
        return
    states, actions, rewards, n_states, dones, _ = mem.get_batch(batch_size)
    targets = policy_net(n_states)
    targets = rewards + GAMMA * np.max(targets, axis=-1)
    targets[dones] = rewards[dones]

    if setting['algo'] == 'optimistic':
        v = policy_net.embeded_vector(state_batch)
        b = policy_net.bonus(v)
        writer.add_scalar('dqn/bonus', torch.mean(b), t)

    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    writer.add_scalar('dqn/Q', torch.mean(next_state_values), t)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss


def training():
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    reward_track = []
    timestep = []
    terminal = True
    state = env.reset()
    state = torch.tensor([state], device=device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    optimizer = optim.SGD(policy_net.parameters(), lr=0.01)
    optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025)
    #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.9993**x)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9992)
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    memory = Memory(100000)
    ep_reward = 0

    max_position=-1
    for t in trange(setting['step']):
        if setting['algo'] == 'greedy':
            action = epsilon_greedy(policy_net, state, t)
        elif setting['algo'] == 'no-explore':
            action = no_explore(policy_net, state)
        elif setting['algo'] == 'optimistic':
            action = no_explore(policy_net, state)
            action_index = action.item()

            x = policy_net.embeded_vector(state)
            #policy_net.cov[action_index] += x.T.mm(x)
            #policy_net.inv_cov[action_index] = torch.inverse(policy_net.cov[action_index])
            A = policy_net.inv_cov[action_index]
            policy_net.inv_cov[action_index] -= A.mm(x.T).mm(x.mm(A)) / (1. + x.mm(A).mm(x.T))

        action_index = action.item()
        next_state, reward, terminal, _ = env.step(action_index)
        max_position = max(max_position, next_state[0])
        if setting['render']:
            env.render()
        ep_reward += reward
        reward = torch.tensor([reward], device=device)
        next_state = None if terminal else torch.tensor([next_state], device=device)
        memory.push(state, action, next_state, reward)

        if t > setting['batch_size']:
            loss = optimize_model(policy_net, target_net, optimizer, memory, t)
            writer.add_scalar('dqn/Loss', loss, t)
            writer.add_scalar('dqn/lr', optimizer.param_groups[0]['lr'], t)
            #scheduler.step()
        state = next_state
        if t % setting['target_update'] == 0:
            target_net.load_state_dict(policy_net.state_dict())
            if setting['algo'] == 'optimistic':
                target_net.inv_cov = [e.detach().clone() for e in policy_net.inv_cov]
        if terminal:
            state = env.reset()
            state = torch.tensor([state], device=device)
            reward_track.append(ep_reward)
            timestep.append(t)
            writer.add_scalar('dqn/reward', ep_reward, t)
            max_position = -1
            ep_reward = 0

    return reward_track, timestep


os.makedirs(setting['saved_dir'], exist_ok=True)
run_time = []
for t in range(setting['start_index'], setting['start_index'] + setting['repeat']):
    start = timeit.default_timer()
    reward_track, timestep = training()
    with open(path.join(setting['saved_dir'], 'result{}.pkl'.format(t)), 'wb') as f:
        pickle.dump([reward_track, timestep], f)
    stop = timeit.default_timer()
    run_time.append('round:{}, {} s.'.format(t, stop - start))
    print('\n'.join(run_time))

env.close()
