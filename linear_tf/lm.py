import pickle
import torch
from .trajectory import Trajectory


class LeastSquare:
    def __init__(self, setting, device):
        self.w = torch.zeros(setting['feature_size'], 1)
        self.t = 0
        self.beta = setting['beta']
        self.device = device
        self.trajectory = Trajectory(setting, device, setting['buffer_size'])

    def reset_w(self):
        self.w.fill_(0)

    def bonus(self, beta, x):
        inv_cov = self.trajectory.inv_cov
        inv_cov = inv_cov.to(self.device) if x.is_cuda else inv_cov
        b = beta * torch.sqrt(x.mm(inv_cov).mm(x.T).diagonal())#.view(-1, 1)
        return b

    def predict(self, x, use_bonus):
        w = self.w.to(self.device) if x.is_cuda else self.w
        Q = x.mm(w)
        if use_bonus:
            B = self.bonus(self.beta, x).view(-1, 1)
            assert Q.shape == B.shape
            Q += B
        return Q

    def convert_to_cpu(self):
        self.w = self.w.cpu()
        self.cov = self.cov.cpu()
        self.inv_cov = self.inv_cov.cpu()

    def fit_(self, X, y):
        return self.trajectory.last_inv_cov.to(self.device).mm(X.T.mm(y)).to('cpu')

    def fit(self, X, y):
        self.w = self.fit_(X, y)
        self.trajectory.inv_cov = self.trajectory.last_inv_cov.clone()

    def smooth_fit(self, X, y):
        self.w = self.w * 0.9 + 0.1 * self.fit_(X, y)

class Model:
    def __init__(self, setting, device):
        self.action_model = [LeastSquare(setting, device)\
                for _ in range(setting['n_action'])]
        self.D = setting['feature_size']
        self.H = setting['step'] ** (1./4)

    def Q(self, state, use_bonus):
        q = [m.predict(state, use_bonus) for m in self.action_model]
        q = torch.stack(q, dim=1).squeeze(2)
        return q

    def choose_action(self, state, use_bonus):
        Q_next = self.Q(state, use_bonus)
        _, index = torch.max(Q_next, dim=1)
        return index

    def update1(self, ls_model, kargs, reward, state, terminal, V_next):
        Q = reward + V_next * (1 - terminal)# - torch.mean(reward)
        ls_model.fit(state, Q)
        assert Q.shape[1] == 1


    def update2(self, ls_model, kargs, reward, state, terminal, V_next):
        Q = reward + V_next - torch.mean(reward)
        ls_model.fit(state, Q)
        assert Q.shape[1] == 1

    def average_reward_algorithm(self, **kargs):
        setting = kargs['setting']
        assert len(kargs['policy']) == setting['n_action']
        for ls_model, action_policy in zip(self.action_model, kargs['policy']):
            state, reward, next_state, terminal = ls_model.trajectory.get_past_data()
            if state.shape[0] == 0:
                continue
            reward = reward.view(-1, 1)
            terminal = terminal.view(-1, 1)
            Q_next = self.Q(next_state, kargs['bonus'])
            if action_policy != None:
                V_next = Q_next.gather(1, action_policy.view(-1, 1))
            else:
                V_next = Q_next.max(dim=1)[0].view(-1, 1)
            if setting['env'] == 'Acrobot-v1':
                #V_next = torch.clamp(V_next , min=-self.H, max=0)
                V_next = torch.clamp(V_next , min=-500, max=0)
            else:
                V_next = torch.clamp(V_next , max=self.H)


            if action_policy != None:
                self.update2(ls_model, kargs, reward, state, terminal,  V_next)
            else:
                if setting['inf_hor']:
                    self.update2(ls_model, kargs, reward, state, terminal, V_next)
                else:
                    self.update1(ls_model, kargs, reward, state, terminal, V_next)

            assert ls_model.w.shape == (self.D, 1)


    def load(self, path):
        with open(path, 'rb') as f:
            tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict)

    def save(self, path):
        with open(path, 'wb') as f:
            self.clear_trajectory()
            pickle.dump(self.__dict__, f)
