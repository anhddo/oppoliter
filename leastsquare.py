import argparse
import os
from os import path
import torch
from linear.fourier_transform import FourierTransform
from linear.lm import Model
from linear.trajectory import Trajectory
from linear.leastsquare_qlearning import LeastSquareQLearning
from linear.politex import Politex
from linear.env import EnvWrapper
import pickle
import timeit
from tqdm import tqdm, trange

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--fourier-order", type=int, default=4)
    parser.add_argument("--algo", default='val')
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--min-epsilon", type=float, default=0.1)
    parser.add_argument("--ep-decay", type=float, default=0.999)
    parser.add_argument("--n-eval", type=int, default=5)
    parser.add_argument("--sample-len", type=int, default=5)
    parser.add_argument("--use-nn", action='store_true')
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--buffer-size", type=int, default=5000)
    parser.add_argument("--tau", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--env", default='CartPole-v0')
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--bonus", action='store_true')
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--save-dir")
    args = parser.parse_args()
    setting = vars(args)
    setting['bonus'] = True if setting['bonus'] == 1 else False


    assert setting['algo'] in ['val', 'pol', 'egreedy', 'politex']

    torch.set_default_tensor_type(torch.DoubleTensor)

    parent_dir = setting['save_dir']
    os.makedirs(parent_dir, exist_ok=True)
    setting['model_path'] = path.join(parent_dir, 'model.pkl')

    with open(path.join(parent_dir, 'setting.txt'), 'w') as f:
        f.write(str(setting))

    for i in trange(setting['start_index'], setting['start_index'] + setting['repeat']):
        if setting['algo'] == 'politex':
            Politex().train(i, setting)
        else:
            LeastSquareQLearning().train(i, setting)

