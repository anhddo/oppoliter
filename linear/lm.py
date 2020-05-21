import pickle
import torch



class LeastSquareModel(object):
    def __init__(self, D, beta, device):
        self.w = torch.zeros(D, 1)
        #self.w = torch.rand(D, 1, dtype=torch.double, device=device) * 2 - 2
        #self.cov = 
        self.inv_cov = 1e6 * torch.eye(self.w.shape[0])
        self.beta = beta
        self.device = device

    def reset_zeros(self):
        self.w.fill_(0)


    def predict(self, x, bonus=True):
        w = self.w.to(self.device) if x.is_cuda else self.w
        Q = x.mm(w)
        if bonus:
            b = self.bonus(x)
            Q = Q + b
            assert Q.shape == b.shape
        return Q

    def bonus(self, x):
        inv_cov = self.inv_cov.to(self.device) if x.is_cuda else self.inv_cov
        v = self.beta * torch.sqrt(x.mm(inv_cov).mm(x.T).diagonal())
        return v.view(-1, 1)

    def update_cov(self, x):
        A = self.inv_cov
        self.inv_cov -= A.mm(x.T).mm(x.mm(A)) / (1. + x.mm(A).mm(x.T))

    def convert_to_cpu(self):
        self.w = self.w.cpu()
        self.cov = self.cov.cpu()
        self.inv_cov = self.inv_cov.cpu()


class Model:
    def __init__(self, ftr_size, action_space, beta, device):
        self.action_count = [0, 0]
        self.action_model = [LeastSquareModel(ftr_size, beta, device) for _ in range(action_space)]
        self.D = ftr_size
        self.beta = beta

    def choose_action(self, state, bonus=True):
        m = self.predict(state, bonus)
        return torch.argmax(m, axis=1)

    def predict(self, state, bonus=True):
        m = [m.predict(state, bonus) for m in self.action_model]
        m = torch.stack(m, dim=1).squeeze(2)
        return m

    def update(self, trajectory_list, env, discount, device, bonus=False, policy=None):
        policy = policy if policy else [None] * len(self.action_model)
        for ls_model, trajectory, policy_per_action in zip(self.action_model, trajectory_list, policy):
            state, reward, next_state, terminal = trajectory.get_past_data()
            if state.shape[0] == 0:
                continue
            reward = reward.view(-1, 1)
            terminal = terminal.view(-1, 1)
            Q_next = self.predict(next_state, bonus)
            if policy_per_action != None:
                V_next = Q_next.gather(1, policy_per_action.view(-1, 1))
            else:
                V_next = Q_next.max(dim=1)[0].view(-1, 1)
            Q = (reward + discount * V_next) * (1 - terminal)
            Q = torch.clamp(Q, max=env.max_clamp)
            ls_model.w = ls_model.inv_cov.to(device).mm(state.T.mm(Q)).cpu()
            assert Q.shape[1] == 1
            assert ls_model.inv_cov.shape == (self.D, self.D)
            assert ls_model.w.shape == (self.D, 1)

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


