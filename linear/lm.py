import pickle
import torch


class LeastSquare:
    def __init__(self, setting, device):
        #self.w = torch.rand(setting['feature_size'], 1) * 2 - 1
        self.w = torch.zeros(setting['feature_size'], 1)
        self.w = torch.nn.init.normal_(self.w)
        #self.w = torch.ones(setting['feature_size'], 1) * 100
        #if setting['algo'] == 'politex':
        #    self.w = torch.ones(setting['feature_size'], 1)
        self.beta = setting['beta']
        self.device = device
        self.inv_cov = 10 * torch.eye(setting['feature_size'])

    def reset_w(self):
        self.w.fill_(0)

    def bonus(self, beta, x):
        inv_cov = self.inv_cov.to(self.device) if x.is_cuda else self.inv_cov
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

    def update_cov(self, state_t):
        A = self.inv_cov
        d = 1. + state_t.mm(A).mm(state_t.T)
        self.inv_cov -= A.mm(state_t.T).mm(state_t.mm(A)) / d

    def convert_to_cpu(self):
        self.w = self.w.cpu()
        self.cov = self.cov.cpu()
        self.inv_cov = self.inv_cov.cpu()

    def fit_(self, X, y):
        return self.inv_cov.to(self.device).mm(X.T.mm(y)).to('cpu')

    def fit(self, X, y):
        self.w = self.fit_(X, y)

    def smooth_fit(self, X, y):
        self.w = self.w * 0.9 + 0.1 * self.fit_(X, y)

class Model:
    def __init__(self, setting, device):
        self.action_model = [LeastSquare(setting, device)\
                for _ in range(setting['n_action'])]
        self.D = setting['feature_size']

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
        #Q = reward + V_next - torch.mean(reward)
        ls_model.fit(state, Q)
        assert Q.shape[1] == 1

    def update2(self, ls_model, kargs, reward, state, terminal, V_next):
        Q = reward + kargs['discount'] * V_next * (1 - terminal)
        ls_model.fit(state, Q)
        assert Q.shape[1] == 1

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
            V_next = torch.clamp(V_next, min=kargs['env'].min_clamp, max=kargs['env'].max_clamp)

            if action_policy != None:
                self.update1(ls_model, kargs, reward, state, terminal,  V_next)
            else:
                self.update2(ls_model, kargs, reward, state, terminal, V_next)

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
