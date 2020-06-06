import numpy as np
import numpy.random as npr
import torch
import copy
from tqdm import tqdm
from .utils import initialize, print_info
from .lm import Model
from os import path
import pickle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.nn.functional import mse_loss, softmax
import time



class Politex:
    def __init__(self):
        name = "Politex"

    def bonus(self, state, inv_cov, setting):
        inv_cov = inv_cov.to(self.device) if state.is_cuda else inv_cov
        B = setting['beta'] * torch.sqrt(state.mm(inv_cov).mm(state.T).diagonal())
        return B

    def bonus_list_inv_cov(self, state, list_inv_cov, setting):
        B = [self.bonus(state, inv_cov, setting) for inv_cov in list_inv_cov]
        B = torch.stack(B, dim=1)
        B = torch.sum(B, dim=1)
        return B

    def choose_action(self, setting, model, state):
        Q = model.Q(state, setting['bonus'])#.squeeze(0)
        p = softmax(setting['lr'] * Q, dim=1)
        A = torch.multinomial(p, 1)
        return A, p

    def train(self, train_index, setting):
        init = initialize(setting)
        writer = SummaryWriter(log_dir='logs/{}-politex{}'\
                .format(setting['env'], '-optimistic' if setting['bonus'] else '') + str( datetime.now()))
        env, trajectory, model, ftr_transform, device =\
                init['env'], init['trajectory'], init['model'],\
                init['ftr_transform'], init['device']
        self.device = device
        self.writer = writer
        terminal = False
        target_track, time_step = [], []
        state = env.reset()
        state = ftr_transform.transform(state)

        episode_count = 0
        t = -1
        expert = Model(setting, device)

        pbar = tqdm(total=setting['step'], leave=True)
        exploration_bonus_per_action = [[expert.action_model[0].inv_cov.clone()]]*setting['n_action']
        q_sum = 0
        setting['K'] = int(np.sqrt(setting['step']))
        setting['tau'] = int(np.sqrt(setting['step']))
        print(setting)
        for k in range(setting['K']):
            for _ in range(setting['tau']):
                pbar.update()
                #env._env.render()
                if terminal:
                    time_step.append(t)
                    target_track.append(env.tracking_value)
                    writer.add_scalar('ls/q', torch.max(expert.Q(state, setting['bonus'])), t)
                    writer.add_scalar('ls/reward', env.tracking_value, t)
                    writer.add_scalar('ls/t', env.t, t)
                    state = env.reset()
                    state = ftr_transform.transform(state)
                    episode_count += 1

                t += 1
                action, p = self.choose_action(setting, model, state)
                #print(action, p)
                next_state, true_reward, _, terminal, info = env.step(action.item())
                bonus = expert.action_model[action].bonus(setting['beta'], state).item() if setting['bonus'] else 0
                writer.add_scalar('ls/bonus', bonus, t)
                modified_reward = true_reward
                #modified_reward = true_reward + bonus
                writer.add_scalar('ls/modified_reward', modified_reward, t)
                writer.add_scalar('ls/w', torch.max(expert.action_model[0].w), t)
                next_state = ftr_transform.transform(next_state)
                trajectory[action].append(state, modified_reward, next_state, terminal)
                expert.action_model[action].update_cov(state)
                state = next_state
                #writer.add_scalar('ls/reward_raw', modified_reward, t)

            policy = []
            for trajectory_per_action in trajectory:
                _, _, next_state, _ = trajectory_per_action.get_past_data()
                if next_state.shape[0] == 0:
                    policy.append(None)
                else:
                    next_action, _ = self.choose_action(setting, model, next_state)
                    policy.append(next_action)

            n_eval = 0
            #for e in expert.action_model:
            #    pass
            #    #e.reset_w()
            loss = 0
            for _ in range(setting['n_eval']):
                w = [m.w for m in expert.action_model]
                expert.average_reward_algorithm(trajectory=trajectory, env=env,\
                        discount=setting['discount'], bonus=False, policy=policy)
                for o_w, m in zip(w, expert.action_model):
                    loss += mse_loss(o_w, m.w)
            writer.add_scalar('politex/mse_w', loss, t)

            for e, m in zip(expert.action_model, model.action_model):
                m.w += e.w

            #if setting['bonus']:
            #    for inv_cov_list, action_model in zip(exploration_bonus_per_action, expert.action_model):
            #        inv_cov_list.append(action_model.inv_cov.clone().detach())

        pbar.close()
        env.reset()
        #env._env.close()
        ftr_transform.save()
        with open(path.join(setting['save_dir'], 'result{}.pkl'.format(train_index)), 'wb') as f:
            pickle.dump([target_track, time_step], f)
