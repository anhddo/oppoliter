import torch


class Trajectory:
    def __init__(self, D, device, horizon_len=10000):
        self.index = -1
        #horizon_len = 10000
        self.state = torch.zeros(horizon_len, D, dtype=torch.double, device=device)
        self.next_state = torch.zeros(horizon_len, D, dtype=torch.double, device=device)
        self.reward = torch.zeros(horizon_len, dtype=torch.double, device=device)
        self.terminal = torch.zeros(horizon_len, dtype=torch.double, device=device)
        self.horizon_len = horizon_len
        self.last_index = -1

   #     self.inv_cov = 10 * torch.eye(setting['feature_size'])
   #     self.last_inv_cov = 10 * torch.eye(setting['feature_size'])


    def append(self, state, reward, next_state, terminal):
        #self.last_index = min(self.last_index + 1, self.horizon_len)
        self.index = (self.index + 1) % self.horizon_len
        self.state[self.index, :] = state
        self.next_state[self.index, :] = next_state
        self.reward[self.index] = reward
        self.terminal[self.index] = int(terminal)

    #def update_cov(self, state_t, remove=False):
    #    A = self.last_inv_cov
    #    d = 1. + state_t.mm(A).mm(state_t.T)
    #    U = A.mm(state_t.T).mm(state_t.mm(A)) / d
    #    if remove:
    #        self.last_inv_cov += U
    #    else:
    #        self.last_inv_cov -= U

    def get_past_data(self):
        #index = self.last_index + 1
        index = self.index + 1
        return (self.state[:index, :],
                self.reward[:index],
                self.next_state[:index, :],
                self.terminal[:index])

    def reset(self):
        self.index = -1

    def bonus(self, x):
        inv_cov = self.inv_cov.to(self.device) if x.is_cuda else self.inv_cov
        v = self.beta * torch.sqrt(x.mm(inv_cov).mm(x.T).diagonal())
        return v.view(-1, 1)
