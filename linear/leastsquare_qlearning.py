import numpy as np
import numpy.random as npr
import torch
from tqdm import tqdm
from .trajectory import Trajectory
from .fourier_transform import FourierTransform
from .utils import initialize
from os import path
import pickle



class LeastSquareQLearning:
    def __init__(self):
        self.name = "Least square value iteration"

    def train(self, train_index, setting) :
        init = initialize(setting)
        env, trajectory, model, ftr_transform, device =\
                init['env'], init['trajectory'], init['model'],\
                init['ftr_transform'], init['device']
        sum_modified_reward = 0
        terminal = True

        state = None
        episode_count = 0
        env.reset()
        t = -1
        pbar = tqdm(total=setting['horizon_len'], leave=True)
        target_track, time_step = [], []
        while t < setting['horizon_len']:
            for _ in range(setting['sample_len']):
                t += 1
                pbar.update()
                if terminal:
                    time_step.append(t)
                    target_track.append(env.tracking_value)
                    sum_modified_reward = 0
                    state = env.reset()
                    state = ftr_transform.transform(state)
                    episode_count += 1
                action = 0
                if self.algo == EPSILON_GREEDY:
                    if npr.uniform() < self.epsilon:
                        action = npr.randint(self.n_action)
                    else:
                        action = model.choose_action(state, setting['bonus']).cpu().numpy()[0]
                else:
                    action = model.choose_action(state, setting['bonus']).cpu().numpy()[0]
                model.action_model[action].update_cov(state)
                next_state, true_reward, modified_reward, terminal, info = env.step(action)
                sum_modified_reward += modified_reward
                next_state = ftr_transform.transform(next_state)
                trajectory[action].append(state, modified_reward, next_state, terminal)
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
                model.average_reward_algorithm(trajectory, env,\
                        setting['discount'], setting['bonus'], policy, device)
        env.reset()
        pbar.close()
        with open(path.join(setting['save_dir'], 'result{}.pkl'.format(train_index)), 'wb') as f:
            pickle.dump([target_track, time_step], f)



