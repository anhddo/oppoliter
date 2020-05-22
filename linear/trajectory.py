import torch


class Trajectory:
    def __init__(self, D, device, horizon_len=10000):
        self.index = -1
        self.state = torch.zeros(horizon_len, D, dtype=torch.double, device=device)
        self.next_state = torch.zeros(horizon_len, D, dtype=torch.double, device=device)
        self.reward = torch.zeros(horizon_len, dtype=torch.double, device=device)
        self.terminal = torch.zeros(horizon_len, dtype=torch.double, device=device)
        self.horizon_len = horizon_len


    def append(self, state, reward, next_state, terminal):
        self.index = self.index + 1
        self.state[self.index, :] = state
        self.next_state[self.index, :] = next_state
        self.reward[self.index] = reward
        self.terminal[self.index] = int(terminal)

    def get_past_data(self):
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
