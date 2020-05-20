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
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

parser = argparse.ArgumentParser(description="Finite-horizon MDP")
parser.add_argument("--env", default='CartPole-v0')
parser.add_argument("--step", type=int, default=10000)
parser.add_argument("--optimistic", action="store_true")
parser.add_argument("--algo", default='')
parser.add_argument("--render", action="store_true")
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--beta", type=float, default=0.01)
parser.add_argument("--start-index", type=int, default=0)
parser.add_argument("--saved-dir")
args = parser.parse_args()
setting = vars(args)

env = gym.make(setting['env'])
n_actions = env.action_space.n
n_observations = env.observation_space.shape[0]
embeded_vector_size = 8

torch.set_default_tensor_type(torch.DoubleTensor)

# set up matplotlib


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        n = 32
        self.conv1 = nn.Linear(input_dim, n)
        self.conv2 = nn.Linear(n, n)
        self.conv3 = nn.Linear(n, embeded_vector_size)
        self.head = nn.Linear(embeded_vector_size, output_dim)
        self.cov = [torch.eye(embeded_vector_size).to(device) * 1e-5] * n_actions
        self.inv_cov = [torch.inverse(M) for M in self.cov]
        self.state_per_action = [[]] * n_actions

    def observe(self, state, action):
        with torch.no_grad():
            self.state_per_action[action].append(state)
            M = torch.stack(self.state_per_action[action], axis=1).squeeze(0)
            M = self.embeded_vector(M)
            self.cov[action] = M.T.mm(M) + torch.eye(embeded_vector_size).to(device) * 1e-5
            self.inv_cov[action] = torch.inverse(self.cov[action])


    def embeded_vector(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
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
        if setting['optimistic']:
            b = self.bonus(v)
            b = b.detach()
            o += b
            o = torch.clamp(o, max=200)
        return o

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = setting['step'] / 10
TARGET_UPDATE = 50






steps_done = 0


policy_net, target_net = None, None

def no_explore(policy_net, state):
    with torch.no_grad():
        action = policy_net(state).max(1)[1].view(1,1)
        assert action <=n_actions
        return action

def epsilon_greedy(policy_net, state, t):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(- t / (setting['step'] / 10))
    #state = torch.tensor(state).view(-1, n_observations)
    if sample > eps_threshold:
        return no_explore(policy_net, state)
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        assert action <=n_actions
        return action


episode_durations = []



def optimize_model(policy_net, target_net, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


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
    optimizer = optim.Adam(policy_net.parameters())
    optimizer = optim.SGD(policy_net.parameters(), lr=0.01)
    memory = ReplayMemory(50000)
    ep_reward = 0

    max_position=-1
    action = 0
    for t in range(setting['step']):
        if setting['algo'] == 'greedy':
            action = epsilon_greedy(policy_net, state, t)
        elif setting['algo'] == 'no-explore':
            action = no_explore(policy_net, state)
        elif setting['optimistic']:
            #print('optimistic')
            action = no_explore(policy_net, state)
            action_index = action.item()
            policy_net.observe(state, action_index)
        action_index = action.item()
        next_state, reward, terminal, _ = env.step(action_index)
        max_position = max(max_position, next_state[0])
        if setting['render']:
            env.render()
        ep_reward += reward
        reward = torch.tensor([reward], device=device)
        next_state = None if terminal else torch.tensor([next_state], device=device)
        memory.push(state, action, next_state, reward)
        if t > 500:
            optimize_model(policy_net, target_net, optimizer, memory)
        state = next_state
        if t % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            target_net.inv_cov = [e.detach().clone() for e in policy_net.inv_cov]
        if terminal:
            state = env.reset()
            state = torch.tensor([state], device=device)
            reward_track.append(ep_reward)
            timestep.append(t)
            print("time_step:{}, value:{}, max_position:{:2f}".format(t, ep_reward, max_position))
            #o, v = policy_net.q(state)
            #b = policy_net.bonus(v)
            #print(o, b)
            #print(policy_net.inv_cov[0])
            #print(policy_net.cov[0])
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
