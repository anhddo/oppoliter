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
from linear.avg_reward import Model, AverageReward, FeatureTransformer, train


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--n-component", type=int, default=100)
    parser.add_argument("--n-step", type=int, default=10000)
    parser.add_argument("--n-run", type=int, default=1)
    parser.add_argument("--discount", type=int, default=0.999)
    args = parser.parse_args()
    setting = vars(args)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    ftr_transform = FeatureTransformer(observation_space, n_components=setting['n_component'])
    algo = AverageReward()
    model = Model(ftr_transform.dimension, action_space)
    reward = train(env, algo, model, ftr_transform, setting['n_step'])
    parent_dir = path.join('tmp', '_'.join(sys.argv))
    os.makedirs(parent_dir, exist_ok=True)
    model.save(path.join(parent_dir, 'model.pkl'))


