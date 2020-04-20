import matplotlib as mpl

mpl.use("Agg")
import numpy as np
from numpy.linalg import inv
import pickle


class LeastSquareModel(object):
    def __init__(self, D):
        self.w = np.zeros((D, 1))
        #self.w = npr.random((D, 1)) * 2 - 2
        self.reset_covarian()
        self.s = np.zeros((D, 1))

    def reset_sum_vector(self):
        self.s[...] = 0

    def reset_covarian(self):
        self.cov = 1e-3 * np.eye(self.w.shape[0])
        self.inv_cov = inv(self.cov)


    def predict(self, x):
        Q = x.dot(self.w)
        b = self.bonus(x)
        Q = Q + b
        assert Q.shape == b.shape
        Q = Q.reshape(-1, 1)
        return Q

    def bonus(self, x):
        v = np.sqrt(x.dot(self.inv_cov).dot(x.T).diagonal())
        return v.reshape(-1, 1)



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

