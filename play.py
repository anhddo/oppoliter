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


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--path")
    parser.add_argument("--fourier-value", type=int, default=4)
    parser.add_argument("--n-step", type=int, default=10000)
    args = parser.parse_args()
    setting = vars(args)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    #ftr_transform = FeatureTransformer(observation_space, n_components=100)
    ftr_transform = FourierTransform(setting['fourier_value'], observation_space, env)
    #print(ftr_transform.transform([1,-1,3,4]))
    #import pdb; pdb.set_trace();
    #import sys; sys.exit();
    model = Model(0, 0)
    model.load(setting['path'])
    print(model.action_model[0].w)

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


