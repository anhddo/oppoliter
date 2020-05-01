import matplotlib as mpl

mpl.use("Agg")
import pickle
import torch



class LeastSquareModel(object):
    def __init__(self, D, beta, device):
        self.w = torch.zeros(D, 1, dtype=torch.double, device=device)
        #self.w = torch.rand(D, 1, dtype=torch.double, device=device) * 2 - 2
        self.cov = 1e-7 * torch.eye(self.w.shape[0], dtype=torch.double, device=device)
        self.inv_cov = torch.inverse(self.cov)
        self.beta = beta


    def predict(self, x, bonus=True):
        Q = x.mm(self.w)
        if bonus:
            b = self.bonus(x)
            Q = Q + b
            assert Q.shape == b.shape
        return Q

    def bonus(self, x):
        v = self.beta * torch.sqrt(x.mm(self.inv_cov).mm(x.T).diagonal())
        return v.view(-1, 1)

    def update_cov(self, x):
        self.cov += x.T.mm(x)
        self.inv_cov = torch.inverse(self.cov)

    def convert_to_cpu(self):
        self.w = self.w.cpu()
        self.cov = self.cov.cpu()
        self.inv_cov = self.inv_cov.cpu()


class Model:
    def __init__(self, ftr_size, action_space, beta, device):
        self.action_count = [0, 0]
        self.action_model = [LeastSquareModel(ftr_size, beta, device) for _ in range(action_space)]
        self.D = ftr_size

    def choose_action(self, state, bonus=True):
        m = self.predict(state, bonus)
        return torch.argmax(m, axis=1)

    def predict(self, state, bonus=True):
        m = [m.predict(state, bonus) for m in self.action_model]
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
            for lm in self.action_model:
                lm.convert_to_cpu()
            pickle.dump(self.__dict__, f)


