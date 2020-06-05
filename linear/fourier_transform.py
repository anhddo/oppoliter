import numpy as np
import numpy.random as npr
import pickle
import torch
from os import path


class FourierTransform:
    def __init__(self, setting):
        self.path = 'ftr/{}-{}'.format(setting['env'], setting['fourier_order'])
        self.scaler = None
        self.k = None
        self.dimension = None

        s = np.arange(setting['fourier_order'])
        a = [s] * setting['n_observation']
        c = np.meshgrid(*a)
        c = [i.flatten() for i in c]
        self.k = np.stack(c).T
        self.dimension = self.k.shape[0]

        self.ftr_ = np.ones((1, self.dimension))
        self.env_name = setting['env']
        self.min_data, self.max_data = None, None
        if self.env_name == 'Acrobot-v1':
            self.min_data = np.array([-np.pi, -np.pi, -13.0, -22]).reshape(1, -1)
            self.max_data = np.array([+np.pi, +np.pi, +13.0, +22]).reshape(1, -1)
        elif self.env_name in ['CartPole-v0', 'CartPole-v1']:
            self.min_data = np.array([-4.8, -4, -0.41, -4]).reshape(1, -1)
            self.max_data = np.array([+4.8, +4, +0.41, +4]).reshape(1, -1)
        elif self.env_name == 'MountainCar-v0':
            self.min_data = np.array([-1.2, -0.07]).reshape(1, -1)
            self.max_data = np.array([0.6, 0.07]).reshape(1, -1)
        self.range_data = self.max_data - self.min_data



    def transform(self, ftr):
        ftr = ftr.reshape(1, -1)
        ftr = (ftr - self.min_data) / self.range_data
        ftr = np.cos(np.pi * self.k.dot(ftr.T)).reshape(-1, self.dimension)
        return torch.from_numpy(ftr)
