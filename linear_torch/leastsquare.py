import numpy as np
import numpy.random as npr
import torch
from tqdm import tqdm
from .trajectory import Trajectory
from .fourier_transform import FourierTransform
from .utils import initialize, print_info
from .env import env_step, env_reset
from os import path
import pickle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from math import cos
import time
from PIL import Image
import pandas as pd


class QLearning:
    def __init__(self):
        self.name = "Least square value iteration"

    def transform(self, x, setting, ftr_transform, device):
        if setting['use_nn']:
           return ftr_transform.transform(torch.Tensor(x).to(device).detach())
        else:
           return ftr_transform.transform(x)

    def greedy(self, model, env, setting, state):
        if npr.uniform() < setting['min_epsilon']:
            return env.action_space.sample()
        else:
            return model.choose_action(state, False).cpu().numpy()[0]

    def train(self, train_index, setting) :
        init = initialize(setting)
        env, model, ftr_transform, device =\
                init['env'],\
                init['model'],\
                init['ftr_transform'], init['device']

        setting['episode_step'] = 0
        pbar = tqdm(total=setting['step'], leave=True)
        reward_list, time_step = [], []
        writer = SummaryWriter(log_dir='logs/{}-{}'\
                .format(setting['algo'], '-optimistic' if setting['bonus'] else '') \
                + setting['env'] + str( datetime.now()))
        state = self.transform(env_reset(env), setting, ftr_transform, device)
        step, episode_step = -1, 0
        episode_count = 0
        total_reward = 0
        terminal = False
        while step < setting['step']:
            if setting['render']:
                env.render()
            for _ in range(setting['sample_len']):
                step += 1
                if step == setting['step']:
                    break

                if terminal:
                    writer.add_scalar('ls/q', torch.max(model.Q(state, False)), step)
                    writer.add_scalar('ls/reward', total_reward, step)

                    writer.add_scalar('ls/reward', total_reward, step)
                    reward_list.append(total_reward)
                    time_step.append(step)
                    total_reward = 0
                    episode_step = 0

                    state = self.transform(env_reset(env), setting, ftr_transform, device)

                pbar.update()


                action = 0
                if setting['algo'] == 'egreedy':
                    action = self.greedy(model, env, setting, state)
                else:
                    action = model.choose_action(state, setting['bonus']).cpu().numpy()[0]

                episode_step += 1
                next_state, reward, terminal = env_step(env, action, episode_step)
                total_reward += reward
                bonus = model.action_model[action].bonus(setting['beta'], state).item() if setting['bonus'] else 0
                #reward += bonus

                writer.add_scalar('ls/bonus', bonus, step)
                writer.add_scalar('ls/w', torch.max(model.action_model[0].w), step)
                writer.add_scalar('ls/reward_raw', reward, step)

                next_state = self.transform(next_state, setting, ftr_transform, device)
                model.action_model[action].trajectory.append(state, reward, next_state, terminal)
                if step % setting['sample_len'] == 0:
                    model.average_reward_algorithm(
                            env=env,
                            bonus=setting['bonus'],
                            #bonus=False,
                            policy=[None] * setting['n_action'],
                            setting=setting
                        )
                state = next_state

        env.reset()
        env.close()
        pbar.close()

        df = pd.DataFrame(data={'step': time_step, 'reward': reward_list})
        df.to_csv(path.join(setting['save_dir'], 'result{}.csv'.format(train_index)))
