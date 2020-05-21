import numpy as np
import numpy.random as npr
import torch
from tqdm import tqdm

VAL = 0
POL = 1
EPSILON_GREEDY = 2


class LeastSquareQLearning:
    def __init__(self, env, model, ftr_transform,
            trajectory_per_action, setting, device):
        self.name = "Least square value iteration"
        self.regulization_matrix = setting['lambda'] * torch.eye(ftr_transform.dimension)
        self.env = env
        self.ftr_transform = ftr_transform
        self.trajectory_per_action = trajectory_per_action
        self.n_action = len(trajectory_per_action)
        self.model = model
        self.total_step = setting['step']
        self.discount = setting['discount']
        self.render = setting['render']
        self.n_eval = setting['n_eval']
        self.sample_len = setting['sample_len']
        self.bonus = setting['bonus']
        self.epsilon = setting['epsilon']
        self.device = device
        if setting['algo'] == 'val':
            self.algo = VAL
        if setting['algo'] == 'pol':
            self.algo = VAL
        if setting['algo'] == 'ep-gr':
            self.algo = EPSILON_GREEDY



    def train(self, train_index):
        sum_modified_reward = 0
        tracking = [0] * 10
        terminal = True
        target_track, time_step = [], []

        state = None
        episode_count = 0
        self.env.reset()
        t = -1
        pbar = tqdm(total=self.total_step, leave=True)
        while t < self.total_step:
            for _ in range(self.sample_len):
                t += 1
                pbar.update()
                if terminal:
                    time_step.append(t)
                    target_track.append(self.env.tracking_value)
                    sum_modified_reward = 0
                    state = self.env.reset()
                    state = self.ftr_transform.transform(state)
                    episode_count += 1

                action = 0
                if self.algo == EPSILON_GREEDY:
                    if npr.uniform() < self.epsilon:
                        action = npr.randint(self.n_action)
                    else:
                        action = self.model.choose_action(state, self.bonus).cpu().numpy()[0]
                else:
                    action = self.model.choose_action(state, self.bonus).cpu().numpy()[0]
                self.model.action_model[action].update_cov(state)
                next_state, true_reward, modified_reward, terminal, info = self.env.step(action)
                sum_modified_reward += modified_reward
                next_state = self.ftr_transform.transform(next_state)
                self.trajectory_per_action[action].append(state, modified_reward, next_state, terminal)
                state = next_state
                if t == self.total_step:
                    break
            policy = self.next_state_policy() if self.algo == POL else [None] * self.env.action_space

            for _ in range(self.n_eval):
                self.model.update(self.trajectory_per_action, self.env, self.discount, self.device, self.bonus, policy)
        self.env.reset()
        pbar.close()
        return target_track, time_step



    def next_state_policy(self):
        policy = []
        for ls_model, trajectory in zip(self.model.action_model, self.trajectory_per_action):
            state, reward, next_state, terminal = trajectory.get_past_data()
            if state.shape[0] == 0:
                continue
            reward = reward.view(-1, 1)
            terminal = terminal.view(-1, 1)
            Q_next = self.model.predict(next_state, self.bonus)
            V_next, index = torch.max(Q_next, dim=1)
            policy.append(index)
        return policy

