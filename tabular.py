import argparse
import json
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
from tabular.algo.value_iteration import OnlineValueIteration
from tabular.finite_mdp import FiniteMDP


if __name__ == "__main__":
    # seed = 1
    # np.random.seed(seed)
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
    parser.add_argument("--no-save", action="store_true", default=False)
    parser.add_argument("--random-reward", action="store_true", default=True)
    args = parser.parse_args()
    setting = vars(args)




    env = FiniteMDP(setting)

    df = []
    step = int(max(setting["n_episode"] / 100, 1))
    episode_index = np.arange(start=0, stop=setting["n_episode"], step=step)
    for _ in trange(setting["n_run"]):
        algorithm_set = [
            ModelBased(algorithm_type=model_based.POLICY_ITERATION, using_previous_estimate=False),
            OnlineValueIteration(),
        ]
        for algorithm in algorithm_set:
            regret = algorithm.run(setting['c'], setting, env)
            regret = regret[::step]

            cumulative_regret = np.cumsum(regret)
            df.append(
                pd.DataFrame(
                    data={
                        "Algorithm": algorithm.name,
                        "Episode": episode_index,
                        "Cumulative_regret": cumulative_regret,
                    }
                )
            )
    np.set_printoptions(precision=3, suppress=True)
    if setting['no_save']:
        now = datetime.now()
        current_time = now.strftime("%d-%m_%H-%M-%S")
        #img_dir = os.path.join("tmp", current_time)
        img_dir = 'tmp'
        os.makedirs(img_dir, exist_ok=True)

        df = pd.concat(df)
        plt.cla()
        sns_plot = sns.lineplot(
            data=df, x="Episode", y="Cumulative_regret", hue="Algorithm"
        )
        sns_plot.set(xlabel="Episodes", ylabel="Cumulative regret")
        sns_plot.legend()
        plt.title(
            "Action:{}, state:{}, reward-random:{}".format(
                setting["n_action"],
                setting["n_state"],
                setting["random_reward"],
            )
        )
        plt.savefig(path.join(img_dir, "{}.png".format(" ".join(sys.argv[1:]), '.png')))
