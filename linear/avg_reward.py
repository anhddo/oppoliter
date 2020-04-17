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


class FourierTransform:
    def __init__(self, n, d, env):
        self.scaler = self.create_scaler(env)
        s = np.arange(n)
        a = [s]*d
        c = np.meshgrid(*a)
        c = [i.flatten() for i in c]
        self.k = np.stack(c).T
        self.dimension = self.k.shape[0]

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
        return np.cos(np.pi * self.k.dot(ftr.T)).reshape(-1, self.dimension)


class Trajectory:
    def __init__(self, D):
        max_unit = 100000
        self.index = -1
        self.state = np.zeros((max_unit, D))
        self.next_state = np.zeros((max_unit, D))
        self.reward = np.zeros(max_unit)
        self.terminal = np.zeros(max_unit)
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
        Q = x.dot(self.w)
        b = self.bonus(x)
        Q = Q + b
        assert Q.shape == b.shape
        return Q.reshape(-1, 1)

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
    def __init__(self):
        self.name = "Least square value iteration"

    def update(self, model, trajectory_per_action):
        GAMMA = 1 - 1./1000
        for ls_model, trajectory in zip(model.action_model, trajectory_per_action):
            state, reward, next_state, terminal = trajectory.get_past_data()
            if state.shape[0] == 0:
                continue
            reward = reward.reshape(-1, 1)
            terminal = terminal.reshape(-1, 1)
            Q_next = model.predict(next_state)
            V_next = np.max(Q_next, axis=1).reshape(-1, 1)
            b = ls_model.bonus(state)
            V_next = np.clip(V_next, 0, 200)
            ls_model.cov = state.T.dot(state) + 1e-3 * np.eye(ls_model.cov.shape[0])
            ls_model.inv_cov = inv(ls_model.cov)
            Q = (reward + GAMMA * V_next) * (1 - terminal)
            ls_model.w = ls_model.inv_cov.dot(state.T.dot(Q))
            assert ls_model.cov.shape == (model.D, model.D)
            assert ls_model.inv_cov.shape == (model.D, model.D)
            assert ls_model.w.shape == (model.D, 1)
            assert b.shape[1] == 1
            assert Q.shape[1] == 1
        return np.min(Q), np.max(Q)

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

def train(env, algo, model, ftr_transform, setting):
    trajectory_per_action = [
        Trajectory(ftr_transform.dimension) for _ in model.action_model
    ]
    episode_reward = 0
    rewards = [0] * 10
    terminal = True
    q_min,q_max=222,0
    last_t = 0
    for t in range(setting['n_step']):
        if terminal:
            eval_reward = test(model, env,ftr_transform)
            state = env.reset()
            state = ftr_transform.transform(state)
            rewards.append(episode_reward)
            print(int(np.mean(rewards)), episode_reward, eval_reward, t)
            q_min,q_max=222,0
            episode_reward = 0
            del rewards[0]
            model.save(setting['model_path'])
            last_t = t
        action = model.choose_action(state)[0]
        next_state, reward, terminal, info = env.step(action)
        episode_reward += reward
        next_state = ftr_transform.transform(next_state)
        if terminal:
            reward = t - last_t - 200
        trajectory_per_action[action].append(state, reward, next_state, terminal)
        state = next_state
        qmin, qmax = algo.update(model, trajectory_per_action)
        q_min = min(qmin, q_min)
        q_max = max(qmax, q_max)
    return rewards


