import argparse
from os import path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")
import gym
import numpy as np
from tqdm import trange
import pandas as pd
import seaborn as sns
import sys
from linear.avg_reward import FourierTransform
from linear.linear_model import Model, PolicyIteration, train


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--fourier-value", type=int, default=4)
    parser.add_argument("--n-episode", type=int, default=1000)
    parser.add_argument("--n-component", type=int, default=100)
    parser.add_argument("--n-run", type=int, default=10)
    args = parser.parse_args()
    setting = vars(args)
    algo_set = [PolicyIteration()]
    #algo_set = [ValueIteration(), PolicyIteration()]
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    #ftr_transform = FeatureTransformer(observation_space, n_components=setting['n_component'])
    ftr_transform = FourierTransform(setting['fourier_value'], observation_space, env)
    print('dimension:', ftr_transform.dimension)
    n_episode = setting['n_episode']
    df = []
    for _ in trange(setting['n_run']):
        for algo in algo_set:
            model = Model(ftr_transform.dimension, action_space)
            reward = train(env, algo, model, ftr_transform, n_episode)
            df.append(
                pd.DataFrame(
                    data={
                        "episode": np.arange(n_episode),
                        "reward": reward,
                        "algorithm": [algo.name] * n_episode,
                    }
                )
            )
    df = pd.concat(df)
    sns_plot = sns.lineplot(x="episode", y="reward", hue="algorithm", data=df)
    sns_plot.set(xlabel="Episodes", ylabel="Cumulative regret")
    sns_plot.legend()
    plt.title("Least square plot")
    plt.savefig(path.join('tmp', "{}.png".format('_'.join(sys.argv))))
