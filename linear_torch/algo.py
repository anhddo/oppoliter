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

    for t in range(setting['step']):
        if terminal:
            #eval_reward = test(model, env,ftr_transform)
            state = env.reset()
            state = ftr_transform.transform(state)
            rewards.append(episode_reward)
            print(int(np.mean(rewards)), episode_reward, t)
            print('===avg reward:',int(np.mean(rewards)), 'train reward:', episode_reward,
                    'step:',t, '===')
            reward_track.append(episode_reward)
            time_step.append(t)
            q_min,q_max=222,0
            episode_reward = 0
            del rewards[0]
            #model.save(setting['model_path'])
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


