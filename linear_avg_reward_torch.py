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
from linear.avg_reward_torch import Model, Trajectory, AverageReward, FourierTransform, train
import pickle
import torch


if __name__ == "__main__":
    np.random.seed(0)
    print('version fourier2')
    env = gym.make("CartPole-v0")
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--save-dir-name")
    parser.add_argument("--fourier-value", type=int, default=4)
    parser.add_argument("--n-component", type=int, default=100)
    parser.add_argument("--n-step", type=int, default=10000)
    parser.add_argument("--n-run", type=int, default=1)
    parser.add_argument("--discount", type=int, default=0.999)
    parser.add_argument("--result-file", default='1')
    args = parser.parse_args()
    setting = vars(args)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
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
    os.makedirs(parent_dir, exist_ok=True)
    setting['model_path'] = path.join(parent_dir, 'model.pkl')
    with open(path.join(parent_dir, 'ftr_transform'), 'wb') as f:
        pickle.dump(ftr_transform, f)
    with open(path.join(parent_dir, 'setting.txt'), 'w') as f:
        f.write(str(setting))
    algo = AverageReward(1e-3, ftr_transform.dimension, device)
    reward_track, time_step = train(env, algo, model, ftr_transform, trajectory_per_action, setting)
    model.save(setting['model_path'])
