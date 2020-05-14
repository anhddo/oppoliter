import numpy as np
import numpy.random as npr
import pickle
from sklearn.preprocessing import MinMaxScaler
import torch


class FourierTransform:
    def __init__(self, fourier_order, feature_dim, env_wrapper):
        self.scaler = self.create_scaler(env_wrapper)
        s = np.arange(fourier_order)
        a = [s] * feature_dim
        c = np.meshgrid(*a)
        c = [i.flatten() for i in c]
        self.k = np.stack(c).T
        self.dimension = self.k.shape[0]


    def create_scaler(self, env):
        terminal = True
        observation_space = env.observation_space
        self.observation_space = observation_space
        state_array = np.zeros((50000, observation_space))
        for t in range(state_array.shape[0]):
            if terminal :
                state = env.reset()
            action = env.env.action_space.sample()
            state, true_reward, modified_reward, terminal, info = env.step(action)
            state_array[t] = state
        env.reset()
        scaler = MinMaxScaler()
        scaler.fit(state_array)
        return scaler


    def transform(self, ftr):
        ftr = ftr.reshape(1, -1)
        self.scaler.partial_fit(ftr)
        ftr = self.scaler.transform(ftr)
        ftr = np.cos(np.pi * self.k.dot(ftr.T)).reshape(-1, self.dimension)
        return torch.from_numpy(ftr)#.type(torch.double)
        #return torch.from_numpy(ftr).type(torch.double).to(self.device)
