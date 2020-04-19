import argparse
import os
import pickle
from os import path
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")
import gym
import numpy as np
from tqdm import trange
from datetime import datetime
import pandas as pd
import seaborn as sns
import sys
<<<<<<< HEAD:linear_avg_reward.py
from linear.fourier_transform import FourierTransform
from linear.lm import Model
from linear.trajectory import Trajectory
from linear.algo import AverageReward, train
import pickle
import timeit
=======
from linear.avg_reward_torch import Model, Trajectory, AverageReward, FourierTransform, train
import pickle
import torch
>>>>>>> torch:linear_avg_reward_torch.py


if __name__ == "__main__":
    np.random.seed(0)
    print('version fourier2')
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--save-dir")
    parser.add_argument("--fourier-order", type=int, default=4)
    parser.add_argument("--env-name", default='CartPole-v0')
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--discount", type=int, default=0.999)
    args = parser.parse_args()
    setting = vars(args)
    setting['tmp_dir'] = '/tmp/oppoliter'

    env = gym.make(setting['env_name'])
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
<<<<<<< HEAD:linear_avg_reward.py

    ftr_transform = FourierTransform(setting['fourier_order'], observation_space, env)

    parent_dir = setting['save_dir']
=======
    device = torch.device('cpu')
    #if torch.cuda.is_available():
    #    device = torch.device('cuda')
    #ftr_transform = FeatureTransformer(observation_space, n_components=setting['n_component'])
    ftr_transform = FourierTransform(setting['fourier_value'], observation_space, env, device)
    model = Model(ftr_transform.dimension, action_space, device)

    trajectory_per_action = [
        Trajectory(ftr_transform.dimension, device) for _ in model.action_model
    ]
    parent_dir = path.join('tmp', setting['save_dir_name'])
>>>>>>> torch:linear_avg_reward_torch.py
    os.makedirs(parent_dir, exist_ok=True)
    os.makedirs(setting['tmp_dir'], exist_ok=True)
    setting['model_path'] = path.join(parent_dir, 'model.pkl')

    with open(path.join(parent_dir, 'ftr_transform'), 'wb') as f:
        pickle.dump(ftr_transform, f)

    with open(path.join(parent_dir, 'setting.txt'), 'w') as f:
        f.write(str(setting))
<<<<<<< HEAD:linear_avg_reward.py

    algo = AverageReward()
    run_time = []
    for i in range(setting['repeat']):
        start = timeit.default_timer()
        model = Model(ftr_transform.dimension, action_space)
        reward_track, time_step = train(env, algo, model, ftr_transform, setting)
        model.save(setting['model_path'])
        with open(path.join(parent_dir, 'result{}.pkl'.format(i)), 'wb') as f:
            pickle.dump([reward_track, time_step], f)
        stop = timeit.default_timer()
        run_time.append('round:{}, {} s.'.format(i, stop - start))
        print('Run time:')
        print('\n'.join(run_time))





=======
    algo = AverageReward(1e-3, ftr_transform.dimension, device)
    reward_track, time_step = train(env, algo, model, ftr_transform, trajectory_per_action, setting)
    model.save(setting['model_path'])
