import numpy as np
import numpy.random as npr
import torch
from linear.lm import Model
import copy
from tqdm import tqdm
from scipy.special import softmax

VAL = 0
POL = 1
EPSILON_GREEDY = 2


class Politex:
    def __init__(self, env, model, ftr_transform,
            trajectory_per_action, setting, device):
        self.name = "Politex"
        self.regulization_matrix = setting['lambda'] * torch.eye(ftr_transform.dimension)
        self.env = env
        self.ftr_transform = ftr_transform
        self.trajectory_per_action = trajectory_per_action
        self.n_action = len(trajectory_per_action)
        self.expert = []
        self.total_step = setting['step']
        self.discount = setting['discount']
        self.render = setting['render']
        self.n_eval = setting['n_eval']
        self.sample_len = setting['sample_len']
        self.bonus = setting['bonus']
        self.epsilon = setting['epsilon']
        self.T = setting['T']
        self.lr = setting['lr']
        self.tau = setting['tau']
        self.device = device
        if setting['algo'] == 'val':
            self.algo = VAL
        if setting['algo'] == 'pol':
            self.algo = VAL
        if setting['algo'] == 'ep-gr':
            self.algo = EPSILON_GREEDY

        self.model = Model(ftr_transform.dimension, env.action_space, setting['beta'], device)


    def train(self, train_index):
        sum_modified_reward = 0
        tracking = [0] * 10
        terminal = True
        target_track, time_step = [], []

        state = None
        episode_count = 0
        self.env.reset()
        t = -1

        expert = Model(self.model.action_model[0].w.shape[0],
                self.env.action_space, self.model.beta, self.device)
        pbar = tqdm(total=self.total_step, leave=True)
        for i in range(self.T):
            for _ in range(self.tau):
                t += 1
                #pbar.update()
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

                q = self.model.predict(state, self.bonus).squeeze(0).cpu().numpy()
                d = softmax(-self.lr * q)
                action = np.random.choice(self.env.action_space, 1, p=d)[0]
                self.model.action_count[action]+=1

                self.model.action_model[action].update_cov(state)
                next_state, true_reward, modified_reward, terminal, info = self.env.step(action)
                sum_modified_reward += modified_reward
                next_state = self.ftr_transform.transform(next_state)
                self.trajectory_per_action[action].append(state, modified_reward, next_state, terminal)
                state = next_state

            for e, m in zip(expert.action_model, self.model.action_model):
                e.reset_zeros()
                e.inv_cov = m.inv_cov
            policy = self.next_state_policy()
            #print(self.trajectory_per_action[0].index)
            #print(self.trajectory_per_action[1].index)
            print(q)
            for _ in range(self.n_eval):
                expert.update(self.trajectory_per_action,
                        self.env,
                        self.discount,
                        self.device,
                        self.bonus, policy)
            #print(94,expert.action_model[0].w.T)
            #print(95, expert.action_model[1].w.T)
            #import pdb; pdb.set_trace();
            print(self.model.action_count)


            for e, m in zip(expert.action_model, self.model.action_model):
                m.w += e.w
            print(99,self.model.action_model[0].w.T)

        self.env.reset()
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
