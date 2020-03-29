import argparse
import os
from datetime import datetime
from os import path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import trange

from tabular.algo import model_based
from tabular.algo.model_based import ModelBased
from tabular.algo.model_free import ModelFree
from tabular.finite_mdp import FiniteMDP

def line_95_percent(cummulative_regret, label=None, color='b'):
    m = np.mean(cummulative_regret, axis=0)
    s = np.std(cummulative_regret, axis=0)
    u = m + s * 1.96
    l = m - s * 1.96
    plt.plot(m, label=label, color=color)
    plt.fill_between(range(cummulative_regret.shape[1]), u, l, alpha=0.2, color=color)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--n-episode", type=int, default=1000)
    parser.add_argument("--n-action", type=int, default=2)
    parser.add_argument("--n-step", type=int, default=2)
    parser.add_argument("--state-per-stage", type=int, default=2)
    parser.add_argument("--p", type=int, default=0.05)
    parser.add_argument("--alpha", type=int, default=0.1)
    parser.add_argument("--n-run", type=int, default=10)
    parser.add_argument("--n-pol-eval-step", type=int, default=1)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--random-reward", action="store_true", default=True)
    args = parser.parse_args()
    setting = vars(args)

    env = FiniteMDP(setting)

    step = int(max(setting["n_episode"] / 100, 1))
    episode_index = np.arange(start=0, stop=setting["n_episode"], step=step)
    algorithm_set = [
        ModelBased(algorithm_type=model_based.POLICY_ITERATION, using_previous_estimate=False),
        ModelFree(),
    ]
    algorithm_set[0].name = 'Policy iteration'

    cummulative_regret_track = {}
    for algorithm in algorithm_set:
        cummulative_regret_batch = []
        for _ in trange(setting["n_run"]):
            regret, _ = algorithm.run(setting['c'], setting, env)
            cummulative_regret = np.cumsum(regret)
            cummulative_regret_batch.append(cummulative_regret)
        cummulative_regret_track[algorithm.name] = np.vstack(cummulative_regret_batch)
    np.set_printoptions(precision=3, suppress=True)

    os.makedirs('tmp', exist_ok=True)

    plt.clf()
    colors = ['b', 'r']
    for ((name, cummulative_regret), color) in zip(cummulative_regret_track.items(), colors):
        line_95_percent(cummulative_regret, name, color)
    plt.xlabel('Episodes')
    plt.ylabel('Cummulative regret')
    plt.legend()
    plt.show()
    plt.savefig(path.join('tmp', "{}.png".format(" ".join(sys.argv[1:]), '.png')))
