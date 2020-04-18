import argparse
import os
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
from linear.avg_reward import Model, AverageReward, FourierTransform, FeatureTransformer, train
import pickle


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--path")
    parser.add_argument("--fourier-range", type=int, default=4)
    parser.add_argument("--n-step", type=int, default=10000)
    args = parser.parse_args()
    setting = vars(args)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    #ftr_transform = FourierTransform(setting['fourier_range'], observation_space, env)
    model = Model(0, 0)
    model.load(path.join(setting['path'], 'model.pkl'))
    ftr_transform = None
    with open(path.join(setting['path'], 'ftr_transform'), 'rb') as f:
        ftr_transform = pickle.load(f)

    episode_reward = 0
    terminal = True
    while True:
        if terminal:
            state = env.reset()
            state = ftr_transform.transform(state)
            print(episode_reward)
            episode_reward = 0
        env.render()
        action = model.choose_action(state)[0]
        next_state, reward, terminal, info = env.step(action)
        episode_reward += reward
        next_state = ftr_transform.transform(next_state)
        state = next_state


