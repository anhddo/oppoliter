import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import torch



class AverageReward:
    def __init__(self, env, model, ftr_transform, trajectory_per_action, setting, device):
        self.name = "Least square value iteration"
        self.regulization_matrix = 1e-4 * torch.eye(ftr_transform.dimension, device=device)
        self.env = env
        self.ftr_transform = ftr_transform
        self.trajectory_per_action = trajectory_per_action
        self.total_step = setting['step']
        self.discount = setting['discount']
        self.model = model
        self.render = setting['render']


    def train(self):
        episode_reward = 0
        sum_modified_reward = 0
        rewards = [0] * 10
        terminal = True
        reward_track, time_step = [], []

        state=None
        for t in range(self.total_step):
            if self.render:
                self.env.env.render()
            if terminal:
                state = self.env.reset()
                state = self.ftr_transform.transform(state)
                rewards.append(episode_reward)
                print('===true reward:', episode_reward,
                        'modified reward:', sum_modified_reward,
                        'step:',t, '===')
                reward_track.append(episode_reward)
                time_step.append(t)
                episode_reward, sum_modified_reward = 0, 0
                del rewards[0]
            action = self.model.choose_action(state).cpu().numpy()[0]
            next_state, true_reward, modified_reward, terminal, info = self.env.step(action)
            episode_reward += true_reward
            sum_modified_reward += modified_reward
            next_state = self.ftr_transform.transform(next_state)
            self.trajectory_per_action[action].append(state, modified_reward, next_state, terminal)
            state = next_state
            self.update_model()
        return reward_track, time_step


    def update_model(self):
        for ls_model, trajectory in zip(self.model.action_model, self.trajectory_per_action):
            state, reward, next_state, terminal = trajectory.get_past_data()
            if state.shape[0] == 0:
                continue
            reward = reward.view(-1, 1)
            terminal = terminal.view(-1, 1)
            Q_next = self.model.predict(next_state)

            V_next, _ = torch.max(Q_next, dim=1)
            b = ls_model.bonus(state)
            V_next = torch.clamp(V_next, self.env.min_clamp, self.env.max_clamp).view(-1, 1)
            ls_model.cov = state.T.mm(state) + self.regulization_matrix
            ls_model.inv_cov = torch.inverse(ls_model.cov)
            Q = (reward + self.discount * V_next) * (1 - terminal)
            ls_model.w = ls_model.inv_cov.mm(state.T.mm(Q))
            assert ls_model.cov.shape == (self.model.D, self.model.D)
            assert ls_model.inv_cov.shape == (self.model.D, self.model.D)
            assert ls_model.w.shape == (self.model.D, 1)
            assert b.shape[1] == 1
            assert Q.shape[1] == 1

def test(model, env, ftr_transform):
    episode_reward = 0
    state = env.reset()
    state = ftr_transform.transform(state)
    while True:
        action = model.choose_action(state)[0]
        next_state, reward, terminal, info = env.step(action)
        episode_reward += reward
        next_state = ftr_transform.transform(next_state)
        state = next_state
        if terminal:
            return episode_reward

