import matplotlib as mpl

mpl.use("Agg")
import pickle
import torch




class LeastSquareModel(object):
    def __init__(self, D, device):
        #self.w = np.zeros((D, 1))
        self.w = torch.rand(D, 1, dtype=torch.double, device=device) * 2 - 2
        self.s = torch.zeros(D, 1, dtype=torch.double, device=device)
        self.cov = 1e-3 * torch.eye(self.w.shape[0], dtype=torch.double, device=device)
        self.inv_cov = torch.inverse(self.cov)


    def predict(self, x):
        Q = x.mm(self.w)
        b = self.bonus(x)
        Q = Q + b
        assert Q.shape == b.shape
        return Q

    def bonus(self, x):
        v = torch.sqrt(x.mm(self.inv_cov).mm(x.T).diagonal())
        return v.view(-1, 1)



class Model:
    def __init__(self, ftr_size, action_space, device):
        self.action_count = [0, 0]
        self.action_model = [LeastSquareModel(ftr_size, device) for _ in range(action_space)]
        self.D = ftr_size

    def choose_action(self, state):
        m = self.predict(state)
        return torch.argmax(m, axis=1)

    def predict(self, state):
        m = [m.predict(state) for m in self.action_model]
        m = torch.stack(m, dim=1).squeeze(2)
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


