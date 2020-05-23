import pickle
import torch


class LeastSquare:
    def __init__(self, feature_size, beta, device):
        self.w = torch.zeros(feature_size, 1)
        self.beta = beta
        self.device = device
        self.inv_cov = 1e4 * torch.eye(feature_size)

    def reset_w(self):
        self.w.fill_(0)

    def reset_cov(self):
        self.inv_cov = 1e4 * torch.eye(self.inv_cov.shape[0])

    def predict(self, x):
        w = self.w.to(self.device) if x.is_cuda else self.w
        Q = x.mm(w)
        return Q

    def bonus(self, x):
        inv_cov = self.inv_cov.to(self.device) if x.is_cuda else self.inv_cov
        return self.beta * torch.sqrt(x.mm(inv_cov).mm(x.T).diagonal())

    def update_cov(self, state):
        A = self.inv_cov
        c = A.mm(state.T).mm(state.mm(A)) / (1. + state.mm(A).mm(state.T))
        self.inv_cov -= c

    def convert_to_cpu(self):
        self.w = self.w.cpu()
        self.cov = self.cov.cpu()
        self.inv_cov = self.inv_cov.cpu()

    def fit(self, X, y):
        self.w = self.inv_cov.to(self.device).mm(X.T.mm(y)).to('cpu')


class Model:
    def __init__(self, setting, device):
        self.action_model = [LeastSquare(setting['feature_size'], setting['beta'], device)\
                for _ in range(setting['n_action'])]
        self.D = setting['feature_size']

    def Q(self, state, use_bonus):
        q = [m.predict(state) for m in self.action_model]
        q = torch.stack(q, dim=1).squeeze(2)
        if use_bonus:
            b = [m.bonus(state) for m in self.action_model]
            b = torch.stack(b, dim=1)
            q += b
        return q

    def choose_action(self, state, use_bonus):
        Q_next = self.Q(state, use_bonus)
        _, index = torch.max(Q_next, dim=1)
        return index

    def average_reward_algorithm(self, **kargs):
        assert len(kargs['policy']) == kargs['env'].action_space
        for ls_model, action_trajectory, action_policy in zip(self.action_model, kargs['trajectory'], kargs['policy']):
            state, reward, next_state, terminal = action_trajectory.get_past_data()
            if state.shape[0] == 0:
                continue
            reward = reward.view(-1, 1)
            terminal = terminal.view(-1, 1)
            Q_next = self.Q(next_state, kargs['bonus'])
            if action_policy != None:
                V_next = Q_next.gather(1, action_policy.view(-1, 1))
            else:
                V_next = Q_next.max(dim=1)[0].view(-1, 1)
            Q = (reward + kargs['discount'] * V_next) * (1 - terminal)
            Q = torch.clamp(Q, min=kargs['env'].min_clamp, max=kargs['env'].max_clamp)
            ls_model.fit(state, Q)
            assert Q.shape[1] == 1
            assert ls_model.inv_cov.shape == (self.D, self.D)
            assert ls_model.w.shape == (self.D, 1)


    def load(self, path):
        with open(path, 'rb') as f:
            tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict)

    def save(self, path):
        with open(path, 'wb') as f:
            self.clear_trajectory()
            pickle.dump(self.__dict__, f)
