import os
from os import path
import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion

mpl.use("Agg")

import numpy as np
from numpy.linalg import inv
from tqdm import trange
from datetime import datetime


class FeatureTransformer:

    """
    FeatureTransformer class:
    Arguments:- 
      env = Environment
      n_components = Number of components each RBFSampler will contain
      samples = Amount of training samples to generate
    """

    def __init__(self, observation_space, n_components=100):
        train_states = np.random.random((20000, observation_space)) * 2 - 2
        scaler = StandardScaler()
        scaler.fit(train_states)
        featurizer = FeatureUnion(
            [(str(i), RBFSampler(1, n_components)) for i in range(observation_space)]
        )
        train_features = featurizer.fit_transform(scaler.transform(train_states))
        self.dimension = train_features.shape[1]
        self.featurizer = featurizer
        self.scaler = scaler

    def transform(self, state):
        scaled_state = self.scaler.transform(np.atleast_2d(state))
        return self.featurizer.transform(scaled_state)


class Trajectory:
    def __init__(self, D):
        max_unit = 50000
        self.index = -1
        self.state = np.zeros((max_unit, D))
        self.next_state = np.zeros((max_unit, D))
        self.reward = np.zeros(max_unit)
        self.terminal = np.zeros(max_unit)
        self.max_unit = max_unit

    def append(self, state, reward, next_state, terminal):
        self.index = (self.index + 1) % self.max_unit
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
    def __init__(self, D):
        self.w = np.zeros((D, 1))
        # self.w = npr.random((D, 1)) * 2 - 2
        self.reset_covarian()
        self.s = np.zeros((D, 1))
        self.trajectories = Trajectory(D)

    def reset_sum_vector(self):
        self.s[...] = 0

    def reset_covarian(self):
        self.cov = 1e-3 * np.eye(self.w.shape[0])
        self.inv_cov = inv(self.cov)

    def reset_trajectory(self):
        self.trajectories.reset()

    def predict(self, x):
        q = x.dot(self.w)
        b = self.bonus(x)
        y = q + b
        assert q.shape == b.shape
        return y.reshape(-1, 1)

    def bonus(self, x):
        v = np.sqrt(x.dot(self.inv_cov).dot(x.T).diagonal())
        return v.reshape(-1, 1)

    def append(self, state, reward, next_state, terminal):
        self.trajectories.append(state, reward, next_state, terminal)


class Model:
    def __init__(self, ftr_size, action_space):
        self.action_count = [0, 0]
        self.action_model = [LeastSquareModel(ftr_size) for _ in range(action_space)]
        self.D = ftr_size

    def choose_action(self, state):
        m = self.predict(state)
        return np.argmax(m, axis=1).flatten()

    def predict(self, state):
        m = [m.predict(state) for m in self.action_model]
        m = np.hstack(m)
        return m


class ValueIteration:
    def __init__(self):
        self.name = "Least square value iteration"

    def update(self, model, trajectory_per_action):
        Y = 0
        for ls_model, trajectory in zip(model.action_model, trajectory_per_action):
            state, reward, next_state, terminal = trajectory.get_past_data()
            if state.shape[0] == 0:
                continue
            reward = reward.reshape(-1, 1)
            terminal = terminal.reshape(-1, 1)
            y = model.predict(next_state)
            y = np.max(y, axis=1).reshape(-1, 1)
            y = reward + y * (1 - terminal)
            b = ls_model.bonus(state)
            y = y - b
            Y = y
            y = np.clip(y, 0, 200)
            ls_model.cov = state.T.dot(state) + 1e-3 * np.eye(ls_model.cov.shape[0])
            ls_model.inv_cov = inv(ls_model.cov)
            ls_model.w = ls_model.inv_cov.dot(state.T.dot(y))
            assert ls_model.cov.shape == (model.D, model.D)
            assert ls_model.inv_cov.shape == (model.D, model.D)
            assert ls_model.w.shape == (model.D, 1)
            assert b.shape[1] == 1
            assert y.shape[1] == 1


class PolicyIteration:
    def __init__(self):
        self.name = "Least square policy iteration"

    def update(self, model, trajectory_per_action):
        y = 0
        next_estimate_action = []
        for ls_model, trajectory in zip(model.action_model, trajectory_per_action):
            state, reward, next_state, terminal = trajectory.get_past_data()
            next_action = np.array(model.choose_action(next_state))
            next_estimate_action.append(next_action)

        for ls_model, trajectory, next_action in zip(
            model.action_model, trajectory_per_action, next_estimate_action
        ):
            state, reward, next_state, terminal = trajectory.get_past_data()
            if state.shape[0] == 0:
                continue
            reward = reward.reshape(-1, 1)
            terminal = terminal.reshape(-1, 1)
            y = model.predict(next_state)
            y = y[np.arange(y.shape[0]), next_action]
            y = y.reshape(-1, 1)
            y = reward + y * (1 - terminal)
            b = ls_model.bonus(state)
            y = y - b
            y = np.clip(y, 0, 200)
            ls_model.cov = state.T.dot(state) + 1e-3 * np.eye(ls_model.cov.shape[0])
            ls_model.inv_cov = inv(ls_model.cov)
            ls_model.w = ls_model.inv_cov.dot(state.T.dot(y))
            assert ls_model.cov.shape == (model.D, model.D)
            assert ls_model.inv_cov.shape == (model.D, model.D)
            assert ls_model.w.shape == (model.D, 1)
            assert b.shape[1] == 1
            assert y.shape[1] == 1


def train(env, algo, model, ftr_transform, n_episode):
    trajectory_per_action = [
        Trajectory(ftr_transform.dimension) for _ in model.action_model
    ]
    best_reward = 0
    rewards = []
    for episode in trange(n_episode):
        state = env.reset()
        state = ftr_transform.transform(state)
        terminal = False
        episode_reward = 0
        while not terminal:
            action = model.choose_action(state)[0]
            next_state, reward, terminal, info = env.step(action)
            episode_reward += reward
            next_state = ftr_transform.transform(next_state)
            trajectory_per_action[action].append(state, reward, next_state, terminal)
            state = next_state
        algo.update(model, trajectory_per_action)
        rewards.append(episode_reward)
    return rewards
