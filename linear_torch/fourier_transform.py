import os
from os import path
import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
mpl.use("Agg")
import numpy as np
import numpy.random as npr
from numpy.linalg import inv
from tqdm import trange
from sklearn.pipeline import FeatureUnion
from datetime import datetime
import pickle
from sklearn.preprocessing import MinMaxScaler
import torch


class FourierTransform:
    def __init__(self, n, d, env, device):
        self.scaler = self.create_scaler(env)
        s = np.arange(n)
        a = [s]*d
        c = np.meshgrid(*a)
        c = [i.flatten() for i in c]
        self.k = np.stack(c).T
        self.dimension = self.k.shape[0]
        self.device = device


    def create_scaler(self, env):
        terminal = True
        observation_space = env.observation_space.shape[0]
        self.observation_space = observation_space
        state_array = np.zeros((20000, observation_space))
        for t in range(state_array.shape[0]):
            if terminal :
                state = env.reset()
            action = npr.randint(2)
            state, reward, terminal, info = env.step(action)
            state_array[t] = state
        scaler = MinMaxScaler()
        scaler.fit(state_array)
        return scaler


    def transform(self, ftr):
        ftr = self.scaler.transform(ftr.reshape(-1, self.observation_space))
        ftr = np.cos(np.pi * self.k.dot(ftr.T)).reshape(-1, self.dimension)
        return torch.from_numpy(ftr).type(torch.double).to(self.device)


class Trajectory:
    def __init__(self, D, device):
        max_unit = 100000
        self.index = -1
        self.state = torch.zeros(max_unit, D, dtype=torch.double, device=device)
        self.next_state = torch.zeros(max_unit, D, dtype=torch.double, device=device)
        self.reward = torch.zeros(max_unit, dtype=torch.double, device=device)
        self.terminal = torch.zeros(max_unit, dtype=torch.double, device=device)
        self.max_unit = max_unit

    def append(self, state, reward, next_state, terminal):
        self.index = self.index + 1
        self.state[self.index, :] = state
        self.next_state[self.index, :] = next_state
        self.reward[self.index] = reward
        self.terminal[self.index] = int(terminal)
        assert self.index <= self.max_unit

    def get_past_data(self):
        index = self.index + 1
        state = self.state[:index, :]
        next_state = self.next_state[:index, :]
        reward = self.reward[:index]
        terminal = self.terminal[:index]
        return state, reward, next_state, terminal

    def reset(self):
        self.index = -1


class LeastSquareModel(object):
    def __init__(self, D, device):
        #self.w = np.zeros((D, 1))
        self.w = torch.rand(D, 1, dtype=torch.double, device=device) * 2 - 2
        self.s = torch.zeros(D, 1, dtype=torch.double, device=device)
        self.cov = 1e-3 * torch.eye(self.w.shape[0], dtype=torch.double, device=device)
        self.inv_cov = torch.inverse(self.cov)


    def predict(self, x):
        Q = x.mm(self.w)
        b = self.bonus(x)
        Q = Q + b
        assert Q.shape == b.shape
        return Q

    def bonus(self, x):
        v = torch.sqrt(x.mm(self.inv_cov).mm(x.T).diagonal())
        return v.view(-1, 1)



class Model:
    def __init__(self, ftr_size, action_space, device):
        self.action_count = [0, 0]
        self.action_model = [LeastSquareModel(ftr_size, device) for _ in range(action_space)]
        self.D = ftr_size

    def choose_action(self, state):
        m = self.predict(state)
        return torch.argmax(m, axis=1)

    def predict(self, state):
        m = [m.predict(state) for m in self.action_model]
        m = torch.stack(m, dim=1).squeeze(2)
        return m

    def clear_trajectory(self):
        for lm in self.action_model:
            lm.trajectories = None

    def load(self, path):
        with open(path, 'rb') as f:
            tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict)


    def save(self, path):
        with open(path, 'wb') as f:
            self.clear_trajectory()
            pickle.dump(self.__dict__, f)


class AverageReward:
    def __init__(self, lambda_, n_ftr_space, device):
        self.name = "Least square value iteration"
        self.regulization_matrix = lambda_ * torch.eye(n_ftr_space, device=device)

    def update(self, model, trajectory_per_action):
        GAMMA = 1 - 1./1000
        for ls_model, trajectory in zip(model.action_model, trajectory_per_action):
            state, reward, next_state, terminal = trajectory.get_past_data()
            if state.shape[0] == 0:
                continue
            reward = reward.view(-1, 1)
            terminal = terminal.view(-1, 1)
            Q_next = model.predict(next_state)

            V_next, _ = torch.max(Q_next, dim=1)
            b = ls_model.bonus(state)
            V_next = torch.clamp(V_next, 0, 200).view(-1, 1)
            #print(state.shape, reward.shape, V_next.shape)
            ls_model.cov = state.T.mm(state) + self.regulization_matrix
            ls_model.inv_cov = torch.inverse(ls_model.cov)
            Q = (reward + GAMMA * V_next) * (1 - terminal)
            #print(torch.max(Q), torch.max(V_next), torch.max(reward), GAMMA, torch.max(terminal))
            ls_model.w = ls_model.inv_cov.mm(state.T.mm(Q))
            #print('w ',torch.max(ls_model.w))
            assert ls_model.cov.shape == (model.D, model.D)
            assert ls_model.inv_cov.shape == (model.D, model.D)
            assert ls_model.w.shape == (model.D, 1)
            assert b.shape[1] == 1
            assert Q.shape[1] == 1

def test(model, env, ftr_transform):
    episode_reward = 0
    state = env.reset()
    state = ftr_transform.transform(state)
    while True:
        action = model.choose_action(state)[0]
        next_state, reward, terminal, info = env.step(action)
        episode_reward += reward
        next_state = ftr_transform.transform(next_state)
        state = next_state
        if terminal:
            return episode_reward

def train(env, algo, model, ftr_transform, trajectory_per_action, setting):
    episode_reward = 0
    rewards = [0] * 10
    terminal = True
    q_min,q_max=222,0
    last_t = 0

    reward_track, time_step = [], []

    for t in range(setting['n_step']):
        if terminal:
            #eval_reward = test(model, env,ftr_transform)
            state = env.reset()
            state = ftr_transform.transform(state)
            rewards.append(episode_reward)
            #print(int(np.mean(rewards)), episode_reward, eval_reward, t)
            print(int(np.mean(rewards)), episode_reward, t)
            reward_track.append(episode_reward)
            time_step.append(t)
            q_min,q_max=222,0
            episode_reward = 0
            del rewards[0]
            model.save(setting['model_path'])
            last_t = t
        action = model.choose_action(state).cpu().numpy()[0]
        next_state, reward, terminal, info = env.step(action)
        episode_reward += reward
        next_state = ftr_transform.transform(next_state)
        if terminal:
            reward = t - last_t - 200
        trajectory_per_action[action].append(state, reward, next_state, terminal)
        state = next_state
        algo.update(model, trajectory_per_action)
    return reward_track, time_step


