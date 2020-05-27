import numpy as np
import numpy.random as npr
import pickle
from sklearn.preprocessing import MinMaxScaler
import torch
from os import path


class FourierTransform:
    def __init__(self, setting):
        self.path = 'ftr/{}-{}'.format(setting['env'], setting['fourier_order'])
        self.scaler = None
        self.k = None
        self.dimension = None
        if path.exists(self.path):
            self.load()
        else:
            self.scaler = MinMaxScaler()
            self.scaler.fit(np.zeros((1, setting['n_observation'])))
            s = np.arange(setting['fourier_order'])
            a = [s] * setting['n_observation']
            c = np.meshgrid(*a)
            c = [i.flatten() for i in c]
            self.k = np.stack(c).T
            self.dimension = self.k.shape[0]

        self.update_feature = setting['update_feature']
        #self.dimension += 1
        self.ftr_ = np.ones((1, self.dimension))


    def load(self):
        with open(self.path, 'rb') as f:
            tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict)

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.__dict__, f)


    def transform(self, ftr):
        ftr = ftr.reshape(1, -1)
        if self.update_feature:
            self.scaler.partial_fit(ftr)
        ftr = self.scaler.transform(ftr)

        ftr = np.cos(np.pi * self.k.dot(ftr.T)).reshape(-1, self.dimension)
        return torch.from_numpy(ftr)
