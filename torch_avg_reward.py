import argparse
import os
from os import path
import torch
from linear_torch.fourier_transform import FourierTransform
from linear_torch.lm import Model
from linear_torch.trajectory import Trajectory
from linear_torch.algo import AverageReward
from linear_torch.env import EnvWrapper
import pickle
import timeit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--save-dir")
    parser.add_argument("--fourier-order", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--lambda", type=float, default=1)
    parser.add_argument("--env-name", default='CartPole-v0')
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--discount", type=float, default=0.999)
    args = parser.parse_args()
    setting = vars(args)
    setting['tmp_dir'] = '/tmp/oppoliter'

    env = EnvWrapper(setting['env_name'])
    device = None
    if setting['cpu']:
        device = torch.device('cpu')
    if not setting['cpu'] and torch.cuda.is_available():
        device = torch.device('cuda')

    ftr_transform = FourierTransform(setting['fourier_order'], env.observation_space, env, device)

    parent_dir = setting['save_dir']
    os.makedirs(parent_dir, exist_ok=True)
    os.makedirs(setting['tmp_dir'], exist_ok=True)
    setting['model_path'] = path.join(parent_dir, 'model.pkl')
    print('---------------------')
    print('Pytorch version')
    print('Environment:', env.env_name)
    print('observation_space:', env.observation_space,
            'action_space:', env.action_space,
            'feature dimension:', ftr_transform.dimension)
    print('---------------------')

    with open(path.join(parent_dir, 'ftr_transform'), 'wb') as f:
        pickle.dump(ftr_transform, f)

    with open(path.join(parent_dir, 'setting.txt'), 'w') as f:
        f.write(str(setting))

    run_time = []
    trajectory_per_action = [
        Trajectory(ftr_transform.dimension, device, setting['step'])
            for _ in range(env.action_space)
    ]
    for i in range(setting['repeat']):
        model = Model(ftr_transform.dimension, env.action_space, setting['beta'], device)
        for trajectory in trajectory_per_action:
            trajectory.reset()
        algo = AverageReward(env, model, ftr_transform, trajectory_per_action, setting, device)

        start = timeit.default_timer()
        reward_track, time_step = algo.train()
        model.save(setting['model_path'])
        with open(path.join(parent_dir, 'result{}.pkl'.format(i)), 'wb') as f:
            pickle.dump([reward_track, time_step], f)
        stop = timeit.default_timer()

        run_time.append('round:{}, {} s.'.format(i, stop - start))
        print('Run time:')
        print('\n'.join(run_time))

