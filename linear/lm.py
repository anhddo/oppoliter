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

    def predict(self, x, use_bonus):
        w = self.w.to(self.device) if x.is_cuda else self.w
        Q = x.mm(w)
        if use_bonus:
            inv_cov = self.inv_cov.to(self.device) if x.is_cuda else self.inv_cov
            b = self.beta * torch.sqrt(x.mm(inv_cov).mm(x.T).diagonal()).reshape(Q.shape)
            Q += b
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
        self.w = self.w * 0.8 + 0.2 * self.fit_(X, y)


class Model:
    def __init__(self, setting, device):
        self.action_model = [LeastSquare(setting['feature_size'], setting['beta'], device)\
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

    def average_reward_algorithm_bk(self, **kargs):
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
            Q = reward + kargs['discount'] * V_next * (1 - terminal)
            ls_model.fit(state, Q)
            print(reward.T)
            print(Q.T)
            assert Q.shape[1] == 1
            assert ls_model.inv_cov.shape == (self.D, self.D)
            assert ls_model.w.shape == (self.D, 1)

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
                #import pdb; pdb.set_trace();
            else:
                V_next = Q_next.max(dim=1)[0].view(-1, 1)
            V_next = torch.clamp(V_next, min=kargs['env'].min_clamp, max=kargs['env'].max_clamp)

            if action_policy != None:
                Q = reward + (V_next ) * (1 -  terminal) - torch.mean(reward)
                #Q = reward + kargs['discount'] * V_next * (1 - terminal)
                #ls_model.smooth_fit(state, Q)
                ls_model.fit(state, Q)
                #print(Q.T)
            else:
                Q = reward + kargs['discount'] * V_next * (1 - terminal)
                ls_model.fit(state, Q)

            assert Q.shape[1] == 1
            assert ls_model.inv_cov.shape == (self.D, self.D)
            assert ls_model.w.shape == (self.D, 1)

    def undiscount_average_reward_algorithm(self, **kargs):
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
            #Q = reward + V_next * (1 - terminal) - torch.mean(reward)
            Q = reward + (V_next - torch.mean(reward)) * (1 -  terminal)
            ls_model.smooth_fit(state, Q)
            #print(kargs['env'].min_clamp, kargs['env'].max_clamp)
            #print(reward.T)
            #print(Q.T)
            #print(V_next.T)
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
