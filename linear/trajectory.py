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





class Trajectory:
    def __init__(self, D, max_step):
        max_unit = max_step
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


