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


class Trajectory:
    def __init__(self, D, device, step=10000):
        max_unit = step
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


