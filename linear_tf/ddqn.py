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

        self._hist_St = np.zeros((max_len, state_shape), dtype=state_dtype)
        self._hist_At = np.zeros(max_len, dtype=int)
        self._hist_Rt_1 = np.zeros(max_len, dtype=float)
        self._hist_St_1 = np.zeros((max_len, state_shape), dtype=state_dtype)
        self._hist_done_1 = np.zeros(max_len, dtype=bool)

    def push(self, St, At, Rt_1, St_1, done_1):
        self._hist_St[self._curr_insert_ptr] = St
        self._hist_At[self._curr_insert_ptr] = At
        self._hist_Rt_1[self._curr_insert_ptr] = Rt_1
        self._hist_St_1[self._curr_insert_ptr] = St_1
        self._hist_done_1[self._curr_insert_ptr] = done_1
        self._curr_len = min(self._curr_len + 1, self.max_len)
        self._curr_insert_ptr =  (self._curr_len + 1) % self.max_len

    def __len__(self):
        return self._curr_len

    def get_batch(self, batch_len):
        indices = np.random.randint(low=0, high=self._curr_len, size=batch_len, dtype=int)
        states = np.take(self._hist_St, indices, axis=0)
        actions = np.take(self._hist_At, indices, axis=0)
        rewards_1 = np.take(self._hist_Rt_1, indices, axis=0)
        states_1 = np.take(self._hist_St_1, indices, axis=0)
        dones_1 = np.take(self._hist_done_1, indices, axis=0)
        return states, actions, rewards_1, states_1, dones_1, indices


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, embeded_size, device):
        super(DQN, self).__init__()
        self.device = device
        n = 64
        self.ln1 = nn.Linear(input_dim, n)
        self.ln2 = nn.Linear(n, n)
        self.ln5 = nn.Linear(n, embeded_size)
        self.head = nn.Linear(embeded_size, output_dim)
        #self.inv_cov = [torch.eye(embeded_size).to(device) * 1e5] * n_actions
        #self.inv_cov = [torch.inverse(M) for M in self.cov]
        self.input_dim = input_dim
        nn.init.kaiming_uniform_(self.ln1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.ln2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.ln5.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.head.weight, nonlinearity='relu')

    def transform(self, x):
        x = x.view(-1, self.input_dim).to(self.device)
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = F.relu(self.ln5(x))
        return x

    def q(self, x):
        v = self.transform(x)
        o = self.head(v)
        return o, v


    def forward(self, x):
        o, v = self.q(x)
        return o

def no_explore(policy_net, state):
    with torch.no_grad():
        #import pdb; pdb.set_trace();
        o = policy_net(state)
        #action = (o + policy_net.bonus(v)).max(1)[1].view(1,1)
        action = o.max(1)[1].view(1,1)
        assert action <=n_actions
        return action

def epsilon_greedy(policy_net, state, t):
    global eps_threshold
    sample = random.random()
    eps_threshold = max(EPS_END, eps_threshold * setting['eps_factor'])
    #eps_threshold = EPS_END + (EPS_START - EPS_END) *  math.exp(- t / (setting['step'] / 10))
    #state = torch.tensor(state).view(-1, n_observations)
    writer.add_scalar('dqn/epsilon', eps_threshold, t)
    if sample > eps_threshold:
        return no_explore(policy_net, state)
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        assert action <=n_actions
        return action





def optimize_model(policy_net, target_net, optimizer, memory, t):
    if len(memory) < setting['batch_size']:
        return
    s0, a0, r, s1, dones, _ = memory.get_batch(setting['batch_size'])
    s0 = torch.tensor(s0).to(device)
    a0 = torch.tensor(a0).to(device)
    r = torch.tensor(r).to(device)
    s1 = torch.tensor(s1).to(device)


    action1 = policy_net(s1).max(1)[1].view(-1, 1)
    Q1 = policy_net(s1).gather(1, action1.view(-1, 1)).squeeze(1)
    Q = policy_net(s0).gather(1, a0.view(-1, 1)).squeeze(1)
    Q1 = r + GAMMA * Q1
    Q1[dones] = r[dones]


    loss = F.mse_loss(Q, Q1)
    writer.add_scalar('dqn/Q', torch.mean(Q), t)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss


def training(setting):
    policy_net = DQN(n_observations, n_actions, setting['embeded_size'], device).to(device)
    target_net = DQN(n_observations, n_actions, setting['embeded_size'], device).to(device)
    reward_track = []
    timestep = []
    terminal = True
    state = env.reset()
    #state = env.state
    #state = torch.tensor([state], device=device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    optimizer = optim.SGD(policy_net.parameters(), lr=0.001)
    optimizer = optim.RMSprop(policy_net.parameters(), lr=0.0001)
    #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.9993**x)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9992)
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    memory = Memory(100000, n_observations, np.float)
    ep_reward = 0

    max_position=-1
    for t in trange(setting['step']):
        state0 = torch.tensor(state).to(device)
        if setting['algo'] == 'greedy':
            action = epsilon_greedy(policy_net, state0, t)
        elif setting['algo'] == 'no-explore':
            action = no_explore(policy_net, state0)
        elif setting['algo'] == 'optimistic':
            action = no_explore(policy_net, state0)
            action_index = action.item()
            x = policy_net.transform(state0)
            #policy_net.cov[action_index] += x.T.mm(x)
            #policy_net.inv_cov[action_index] = torch.inverse(policy_net.cov[action_index])
            #A = policy_net.inv_cov[action_index]
            #policy_net.inv_cov[action_index] = 0.99999 * A - A.mm(x.T).mm(x.mm(A)) / (1. + x.mm(A).mm(x.T))
            #policy_net.inv_cov[action_index] = A - A.mm(x.T).mm(x.mm(A)) / (1. + x.mm(A).mm(x.T))
        action_index = action.item()
        next_state, reward, terminal, _ = env.step(action_index)
        #next_state = env.state
        max_position = max(max_position, next_state[0])
        if setting['render']:
            env.render()
        ep_reward += reward
        reward = torch.tensor([reward], device=device)
        #next_state = None if terminal else torch.tensor([next_state], device=device)
        memory.push(state, action, reward, next_state, terminal)

        state = next_state
        if t > setting['batch_size'] * 3:
            loss = optimize_model(policy_net, target_net, optimizer, memory, t)
            writer.add_scalar('dqn/Loss', loss, t)
            writer.add_scalar('dqn/lr', optimizer.param_groups[0]['lr'], t)
            #scheduler.step()
            if t % setting['target_update'] == 0:
                target_net.load_state_dict(policy_net.state_dict())
                #if setting['algo'] == 'optimistic':
                #    target_net.inv_cov = [e.detach().clone() for e in policy_net.inv_cov]
        #print(t)
        #import pdb; pdb.set_trace();
        #if t > 0 and t % 30000 == 0:
        #    policy_net.inv_cov = [torch.eye(setting['embeded_size']).to(device) * 1e5] * n_actions
        #    target_net.inv_cov = [torch.eye(setting['embeded_size']).to(device) * 1e5] * n_actions
        if terminal:
            state = env.reset()
            #state = env.state
            reward_track.append(ep_reward)
            timestep.append(t)
            writer.add_scalar('dqn/reward', ep_reward, t)
            max_position = -1
            ep_reward = 0

    torch.save(policy_net.state_dict(), '../tmp/{}-nn-model'.format(setting['env']))
    return reward_track, timestep

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--env", default='Acrobot-v1')
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
    parser.add_argument("--eps-factor", type=float, default=0.9995)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--saved-dir")
    args = parser.parse_args()
    setting = vars(args)

    writer = SummaryWriter( '../logs/{}-{}-bs{}-ut-{}-{}'.format(
        setting['env'],
        setting['algo'],
        setting['batch_size'],
        setting['target_update'],
        datetime.now()
        ))
    env = gym.make(setting['env'])
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]
    #if setting['env'] == 'Acrobot-v1':
    #    n_observations = 4
    print(n_observations)

    torch.set_default_tensor_type(torch.DoubleTensor)

    device = torch.device("cuda" if torch.cuda.is_available() and not setting['cpu'] else "cpu")



    BATCH_SIZE = setting['batch_size']
    GAMMA = 0.95
    EPS_START = 0.9
    EPS_END = 0.01




    steps_done = 0


    policy_net, target_net = None, None
    eps_threshold=1.
    episode_durations = []

    os.makedirs(setting['saved_dir'], exist_ok=True)
    run_time = []
    for t in range(setting['start_index'], setting['start_index'] + setting['repeat']):
        start = timeit.default_timer()
        reward_track, timestep = training(setting)
        with open(path.join(setting['saved_dir'], 'result{}.pkl'.format(t)), 'wb') as f:
            pickle.dump([reward_track, timestep], f)
        stop = timeit.default_timer()
        run_time.append('round:{}, {} s.'.format(t, stop - start))
        print('\n'.join(run_time))

    env.close()
