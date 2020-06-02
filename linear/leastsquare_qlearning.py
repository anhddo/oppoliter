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
from PIL import Image
import gc


class LeastSquareQLearning:
    def __init__(self):
        self.name = "Least square value iteration"

    def transform(self, x, setting, ftr_transform, device):
        if setting['use_nn']:
           return ftr_transform.transform(torch.Tensor(x).to(device).detach())
        else:
           return ftr_transform.transform(x)

    def train(self, train_index, setting) :
        init = initialize(setting)
        env, model, ftr_transform, device =\
                init['env'],\
                init['model'],\
                init['ftr_transform'], init['device']
        #print_info(setting)
        terminal = False

        print(setting)
        state = None
        episode_count = 0
        t = -1
        pbar = tqdm(total=setting['step'], leave=True)
        target_track, time_step = [], []
        epsilon = 0.8
        writer = SummaryWriter(log_dir='logs/{}-{}'\
                .format(setting['algo'], '-optimistic' if setting['bonus'] else '') \
                + setting['env'] + str( datetime.now()))
        state_ = env.reset()
        state = self.transform(state_, setting, ftr_transform, device)
        #setting['discount'] = 1. - setting['step'] ** (-1. / 4)
        #setting['beta'] = 1. / (1. - setting['discount'])
        #print(setting['discount'], setting['beta'])
        while t < setting['step']:
            if setting['render']:
                env._env.render()
            for _ in range(setting['sample_len']):
                t += 1
                if t == setting['step']:
                    break
                if terminal:
                    #env._env.render()
                    #A = env._env.render(mode='rgb_array')
                    #im = Image.fromarray(A)
                    #im.save('notebook/img/{}-{}.png'.format(env.tracking_value, env.tracking_value))
                    writer.add_scalar('ls/q', torch.max(model.Q(state, False)), t)
                    #print()
                    #print(env.tracking_value)
                    #print(modified_reward)
                    #print()
                    #time.sleep(2)
                    time_step.append(t)
                    target_track.append(env.tracking_value)
                    writer.add_scalar('ls/reward', env.tracking_value, t)
                    writer.add_scalar('ls/t', env.t, t)
                    state_ = env.reset()
                    state = self.transform(state_, setting, ftr_transform, device)
                    torch.cuda.empty_cache()
                pbar.update()
                action = 0
                if setting['algo'] == 'egreedy':
                    #epsilon = max(setting['min_epsilon'], epsilon * setting['ep_decay'])
                    #writer.add_scalar('egreedy/epsilon', epsilon, t)
                    #if npr.uniform() < epsilon:
                    if npr.uniform() < setting['min_epsilon']:
                        action = npr.randint(setting['n_action'])
                    else:
                        action = model.choose_action(state, False).cpu().numpy()[0]
                else:
                    action = model.choose_action(state, setting['bonus']).cpu().numpy()[0]
                next_state, _, modified_reward, terminal, _ = env.step(action)
                bonus = model.action_model[action].bonus(setting['beta'], state).item() if setting['bonus'] else 0
                writer.add_scalar('ls/bonus', bonus, t)
                writer.add_scalar('ls/w', torch.max(model.action_model[0].w), t)
                #modified_reward += bonus
                writer.add_scalar('ls/reward_raw', modified_reward, t)
                next_state = self.transform(next_state, setting, ftr_transform, device)
                #next_state = ftr_transform.transform(next_state)
                #next_state = state
                model.action_model[action].trajectory.append(state, modified_reward, next_state, terminal)
                #model.action_model[action].update_cov(state)
                if t % setting['sample_len'] == 0:
                    model.average_reward_algorithm(
                            env=env,
                            #bonus=setting['bonus'],
                            bonus=setting['bonus'],
                            policy=[None] * setting['n_action'],
                        )
                state = next_state

        env.reset()
        env._env.close()
        pbar.close()
        #ftr_transform.save()
        with open(path.join(setting['save_dir'], 'result{}.pkl'.format(train_index)), 'wb') as f:
            pickle.dump([target_track, time_step], f)
