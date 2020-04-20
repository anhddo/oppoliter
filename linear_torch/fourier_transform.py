import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import numpy.random as npr
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
        observation_space = env.observation_space
        self.observation_space = observation_space
        state_array = np.zeros((50000, observation_space))
        for t in range(state_array.shape[0]):
            if terminal :
                state = env.reset()
            action = npr.randint(2)
            state, true_reward, modified_reward, terminal, info = env.step(action)
            state_array[t] = state
        scaler = MinMaxScaler()
        scaler.fit(state_array)
        return scaler


    def transform(self, ftr):
        ftr = self.scaler.transform(ftr.reshape(-1, self.observation_space))
        ftr = np.cos(np.pi * self.k.dot(ftr.T)).reshape(-1, self.dimension)
        return torch.from_numpy(ftr).type(torch.double).to(self.device)


