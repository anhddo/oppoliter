import numpy as np
import numpy.random as npr
import torch
from tqdm import tqdm
from .trajectory import Trajectory
from .fourier_transform import FourierTransform
from .utils import initialize, print_info
from os import path
import pickle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from math import cos
import time


class LeastSquareQLearning:
    def __init__(self):
        self.name = "Least square value iteration"

    def train(self, train_index, setting) :
        init = initialize(setting)
        env, trajectory, model, ftr_transform, device =\
                init['env'], init['trajectory'], init['model'],\
                init['ftr_transform'], init['device']
        #print_info(setting)
        sum_modified_reward = 0
        terminal = False

        state = None
        episode_count = 0
        t = -1
        pbar = tqdm(total=setting['horizon_len'], leave=True)
        target_track, time_step = [], []
        #setting['discount'] = 1 - setting['horizon_len']**(-1. / 4)
        #setting['beta'] = 1. / (1. - setting['discount'])
        epsilon = 1
        writer = SummaryWriter(log_dir='logs/{}-'.format(setting['algo']) + setting['env'] + str( datetime.now()))
        sum_modified_reward = 0
        state_ = env.reset()
        state = ftr_transform.transform(state_)
        while t < setting['horizon_len']:
            for _ in range(setting['sample_len']):
                if terminal:
                    writer.add_scalar('ls/q', torch.max(model.Q(state, setting['bonus'])), t)
                    time_step.append(t)
                    target_track.append(env.tracking_value)
                    writer.add_scalar('ls/reward', env.tracking_value, t)
                    writer.add_scalar('ls/t', env.t, t)
                    state_ = env.reset()
                    state = ftr_transform.transform(state_)
                t += 1
                pbar.update()
                action = 0
                if setting['algo'] == 'egreedy':
                    epsilon = max(setting['min_epsilon'], epsilon * setting['ep_decay'])
                    writer.add_scalar('egreedy/epsilon', epsilon, t)
                    if npr.uniform() < epsilon:
                        action = npr.randint(setting['n_action'])
                    else:
                        action = model.choose_action(state, setting['bonus']).cpu().numpy()[0]
                else:
                    action = model.choose_action(state, setting['bonus']).cpu().numpy()[0]
                next_state, true_reward, modified_reward, terminal, _ = env.step(action)
                writer.add_scalar('ls/reward_raw', modified_reward, t)
                state_=next_state
                sum_modified_reward += modified_reward
                next_state = ftr_transform.transform(next_state)
                trajectory[action].append(state, modified_reward, next_state, terminal)
                model.action_model[action].update_cov(state)
                state = next_state
                if t == setting['horizon_len']:
                    break

            policy = []
            if setting['algo'] == 'pol':
                for trajectory_per_action in trajectory:
                    _, _, next_state, _ = trajectory_per_action.get_past_data()
                    if next_state.shape[0] == 0:
                        policy.append(None)
                    else:
                        policy.append(model.choose_action(next_state, setting['bonus']))
            else:
                policy = [None] * env.action_space

            for _ in range(setting['n_eval']):
                model.average_reward_algorithm(trajectory=trajectory, env=env,\
                        discount=setting['discount'], bonus=setting['bonus'], policy=policy)
        env.reset()
        pbar.close()
        ftr_transform.save()
        with open(path.join(setting['save_dir'], 'result{}.pkl'.format(train_index)), 'wb') as f:
            pickle.dump([target_track, time_step], f)
