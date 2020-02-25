import argparse
import json
import os
from datetime import datetime
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import trange

from tabular.algo import PolicyIteration, OnlineValueIteration
from tabular.mdp_setting import FiniteMDP

if __name__ == "__main__":
    # np.random.seed(0)
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--n-episode", type=int, default=1000)
    parser.add_argument("--n-action", type=int, default=2)
    parser.add_argument("--n-step", type=int, default=2)
    parser.add_argument("--state-per-stage", type=int, default=2)
    parser.add_argument("--p", type=int, default=0.05)
    parser.add_argument("--alpha", type=int, default=0.1)
    parser.add_argument("--n-run", type=int, default=10)
    args = parser.parse_args()
    setting = vars(args)

    np.set_printoptions(precision=3, suppress=True)

    now = datetime.now()
    current_time = now.strftime("%d-%m_%H-%M-%S")
    img_dir = os.path.join("tmp", current_time)
    os.makedirs(img_dir, exist_ok=True)

    mdp = FiniteMDP(setting)
    env = mdp.env

    plt.cla()
    df = []
    step = int(max(setting["n_episode"] / 100, 1))
    episode_index = np.arange(start=0, stop=setting["n_episode"], step=step)
    for _ in trange(setting['n_run']):
        algorithm_set = [PolicyIteration(setting, env), OnlineValueIteration(setting, env)]
        for algo in algorithm_set:
            regret = algo.run(c=0.1)
            regret = regret[::step]

            cumulative_regret = np.cumsum(regret)
            df.append(
                pd.DataFrame(
                    data={
                        "Algorithm": algo.name,
                        "Episode": episode_index,
                        "Cumulative_regret": cumulative_regret,
                    }
                )
            )
    df = pd.concat(df)
    sns_plot = sns.lineplot(
        data=df, x="Episode", y="Cumulative_regret", hue="Algorithm"
    )
    sns_plot.set(xlabel="Episodes", ylabel="Cumulative regret")
    sns_plot.legend()
    plt.show()

    sns_plot.get_figure().savefig(path.join(img_dir, "plot.png"))
    with open(path.join(img_dir, "setting.json"), "w") as f:
        json.dump(setting, f)
