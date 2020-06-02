import argparse
import os
import sys
from os import path
import pandas as pd
import numpy as np
from tqdm import trange
import numpy.random as npr
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class Env():
    def __init__(self, N):
        self.c_s = 0
        self.N = N


    def step(self, a):
        if a == 0:
            self.c_s -= 1
        if a == 1:
            self.c_s += 1


        r = 0
        if self.c_s == 0:
            r = 0.01
        elif self.c_s == self.N - 1:
            #r = 1
            if npr.uniform() < 0.1:
                r = 1

        if self.c_s < 0:
            self.c_s = 1
        if self.c_s == self.N:
            self.c_s = self.N - 2
        return self.c_s, r

    def reset(self):
        self.c_s = 0
        return 0


class Agent:
    def __init__(self, N, T):
        n_action = 2
        self.H = T**(1. / 4)
        self.B = self.H
        #self.H *= 0.1
        #self.B *= 0.1
        self.g = 1. - 1. / self.B
        #self.Q = np.zeros((N, n_action))
        self.Q = np.ones((N, n_action)) * self.H
        self.N = np.ones((N, n_action))
        self.n_action = n_action
        self.N_ = N
        #print(self.H, self.g)

    def bonus(self, s, a):
        n = self.N[s, a]
        B = self.H if np.abs(n) < 1e-4 else 1. / np.sqrt(n)
        return self.B * B

    def action(self, s):
        bonus = np.array([self.bonus(s, a) for a in range(self.n_action)])
        Q = self.Q[s, :] + bonus
        #if s== self.N_-2:
        #    print(bonus, Q, self.Q[s,:], 'l' if self.Q[s,0] > self.Q[s,1] else 'r')
        a = np.argmax(Q)
        return a

    def V(self, s):
        v =  np.max([self.Q[s,a] + self.bonus(s, a) for a in range(2)])
        v = np.clip(v, None, self.H)
        return v


    def update(self, s, r, a, ns):
        self.N[s, a] += 1
        self.Q[s, a] = r + self.g * self.V(ns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--chain", type=int, default=50)
    parser.add_argument("--step", type=int, default=2000)
    parser.add_argument("--algo")

    args = parser.parse_args()
    setting = vars(args)
    assert setting['algo'] in ['opt', 'greedy', 'noex']
    agent = Agent(setting['chain'], setting['step'])
    writer = SummaryWriter(log_dir='logs/n_chain_{}-{}-{}'\
            .format(setting['algo'], setting['chain'], setting['step']) + str( datetime.now()))

    #df.to_csv('tmp/n_chain/{}_{}_{}'.format(setting['algo'], setting['chain'], setting['step']))

    if setting['algo']=='noex' or setting['algo'] == 'greedy':
        agent.B = 0
    c_s = 1
    N = setting['chain']
    n_action = 2
    env = Env(N)

    reward = []

    state = []
    s = 0

    c_reward = 0
    for t in trange(setting['step']):
        a = agent.action(s)
        ns, r = env.step(a)
        #if s == N -2:
        #    print(s,r, a, ns, t)
        c_reward += r
        agent.update(s, r, a, ns)
        #print(s, r, 'right' if a==1 else 'left', ns)
        writer.add_scalar('chain/reward', c_reward, t)
        writer.add_scalar('chain/Q', agent.Q[s, a], t)
        writer.add_scalar('detail/Q_n-2, 0', agent.Q[N-2, 0], t)
        writer.add_scalar('detail/Q_n-2, 1', agent.Q[N-2, 1], t)
        writer.add_scalar('chain/bonus', agent.bonus(s, a), t)
        writer.add_scalar('chain/state', s, t)
        writer.add_scalar('chain/single reward', r, t)
        reward.append(r)
        state.append(s)
        s = ns
    df = pd.DataFrame({'reward': reward, 'state': state})
    #print(agent.Q)
    df.to_csv('tmp/n_chain/{}_{}_{}'.format(setting['algo'], setting['chain'], setting['step']))
