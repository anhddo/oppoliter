import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import torch



class PolicyIteration:
    def __init__(self, env, model, ftr_transform,
            trajectory_per_action, setting, device):
        self.name = "Least square value iteration"
        self.regulization_matrix = setting['lambda'] * torch.eye(ftr_transform.dimension, device=device)
        self.env = env
        self.ftr_transform = ftr_transform
        self.trajectory_per_action = trajectory_per_action
        self.model = model
        self.total_step = setting['step']
        self.discount = setting['discount']
        self.render = setting['render']
        self.n_eval = setting['n_eval']
        self.sample_len = setting['sample_len']


    def train(self):
        sum_modified_reward = 0
        tracking = [0] * 10
        terminal = True
        target_track, time_step = [], []

        state = None
        episode_count = 0
        self.env.reset()
        t = -1
        while t < self.total_step:
            for _ in range(self.sample_len):
                t += 1
                if terminal:
                    print('==target: {:04.2f}, modified reward: {:04.2f}, step:{:5d}, ep:{:3d}=='
                            .format(self.env.tracking_value, sum_modified_reward, t, episode_count
                        ))
                    time_step.append(t)
                    target_track.append(self.env.tracking_value)
                    sum_modified_reward = 0
                    state = self.env.reset()
                    state = self.ftr_transform.transform(state)
                    episode_count += 1
                action = self.model.choose_action(state).cpu().numpy()[0]
                self.model.action_model[action].update_cov(state)
                next_state, true_reward, modified_reward, terminal, info = self.env.step(action)
                sum_modified_reward += modified_reward
                next_state = self.ftr_transform.transform(next_state)
                self.trajectory_per_action[action].append(state, modified_reward, next_state, terminal)
                state = next_state
                if t == self.total_step:
                    break
            policy = self.get_fix_policy()
            #self.update_cov()

            print('========Policy evaluation======')

            for m in self.model.action_model:
                m.reset_zeros()

            for _ in range(self.n_eval):
                self.update_model(policy)
        self.env.reset()
        return target_track, time_step

    def update_cov(self):
        for ls_model, trajectory in zip(self.model.action_model, self.trajectory_per_action):
            state, _, _, _ = trajectory.get_past_data()
            if state.shape[0] == 0:
                continue
            ls_model.cov = state.T.mm(state) + self.regulization_matrix
            ls_model.inv_cov = torch.inverse(ls_model.cov)


    def get_fix_policy(self):
        policy = []
        for ls_model, trajectory in zip(self.model.action_model, self.trajectory_per_action):
            state, reward, next_state, terminal = trajectory.get_past_data()
            if state.shape[0] == 0:
                continue
            reward = reward.view(-1, 1)
            terminal = terminal.view(-1, 1)
            Q_next = self.model.predict(next_state)
            V_next, index = torch.max(Q_next, dim=1)
            policy.append(index)
        return policy

    def update_model(self, policy):
        for ls_model, trajectory, policy_per_action in zip(self.model.action_model, self.trajectory_per_action, policy):
            state, reward, next_state, terminal = trajectory.get_past_data()
            if state.shape[0] == 0:
                continue
            reward = reward.view(-1, 1)
            terminal = terminal.view(-1, 1)
            Q_next = self.model.predict(next_state)
            Q_next = torch.gather(Q_next, 1, policy_per_action.view(-1, 1))
            Q_next = torch.clamp(Q_next, max=self.env.max_clamp)
            Q = (reward + self.discount * Q_next) * (1 - terminal)
            ls_model.w = ls_model.inv_cov.mm(state.T.mm(Q))
            assert ls_model.cov.shape == (self.model.D, self.model.D)
            assert ls_model.inv_cov.shape == (self.model.D, self.model.D)
            assert ls_model.w.shape == (self.model.D, 1)
            assert Q.shape[1] == 1

