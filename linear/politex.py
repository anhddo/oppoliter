import numpy as np
import numpy.random as npr
import torch
from linear.lm import Model
import copy
from tqdm import tqdm
from scipy.special import softmax
from .utils import initialize
from .lm import Model
from os import path
import pickle


class Politex:
    def __init__(self):
        name = "Politex"

    def train(self, train_index, setting):
        init = initialize(setting)
        env, trajectory, model, ftr_transform, device =\
                init['env'], init['trajectory'], init['model'],\
                init['ftr_transform'], init['device']
        sum_modified_reward = 0
        terminal = True
        target_track, time_step = [], []

        state = None
        episode_count = 0
        env.reset()
        t = -1
        expert = Model(setting, device)

        pbar = tqdm(total=setting['horizon_len'], leave=True)
        exploration_bonus = [[]]*setting['n_action']
        for i in range(setting['T']):
            if setting['on_policy']:
                for e in trajectory:
                    e.reset()
                for e in expert.action_model:
                    pass
                    #e.reset_w()
                    e.reset_cov()
            for _ in range(setting['tau']):
                t += 1
                pbar.update()
                if terminal:
                    time_step.append(t)
                    target_track.append(env.tracking_value)
                    sum_modified_reward = 0
                    state = env.reset()
                    state = ftr_transform.transform(state)
                    episode_count += 1

                q = model.Q(state, False).squeeze(0)
                if setting['bonus']:
                    b = torch.zeros(setting['n_action'])
                    for j in range(setting['n_action']):
                        for inv_cov in exploration_bonus[j]:
                            b[j] += torch.sqrt(state.mm(inv_cov).mm(state.T)).item()
                    q += b
                d = softmax(setting['lr'] * q)
                action = np.random.choice(env.action_space, 1, p=d)[0]

                next_state, true_reward, modified_reward, terminal, info = env.step(action)
                sum_modified_reward += modified_reward
                next_state = ftr_transform.transform(next_state)
                trajectory[action].append(state, modified_reward, next_state, terminal)
                expert.action_model[action].update_cov(state)
                state = next_state

            for e in expert.action_model:
                pass
                e.reset_w()

            policy = []
            for trajectory_per_action in trajectory:
                _, _, next_state, _ = trajectory_per_action.get_past_data()
                if next_state.shape[0] == 0:
                    policy.append(None)
                else:
                    policy.append(model.choose_action(next_state, setting['bonus']))

            for _ in range(setting['n_eval']):
                expert.undiscount_average_reward_algorithm(trajectory=trajectory, env=env,\
                        discount=setting['discount'], bonus=setting['bonus'], policy=policy)

            for e, m in zip(expert.action_model, model.action_model):
                m.w += e.w

            if setting['bonus']:
                for inv_cov_list, action_model in zip(exploration_bonus, expert.action_model):
                    inv_cov_list.append(action_model.inv_cov.clone().detach())

        pbar.close()
        env.reset()
        ftr_transform.save()
        with open(path.join(setting['save_dir'], 'result{}.pkl'.format(train_index)), 'wb') as f:
            pickle.dump([target_track, time_step], f)
