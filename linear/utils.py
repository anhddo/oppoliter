from .trajectory import Trajectory
from .fourier_transform import FourierTransform
from .lm import Model
from .env import EnvWrapper
import torch

def initialize(setting):
    env = EnvWrapper(setting['env'])
    device = get_device(setting)
    ftr_transform = FourierTransform(setting['fourier_order'], env.observation_space, env)
    horizon_len = setting['step']
    if setting['algo'] == 'politex':
        horizon_len = setting['T'] * setting['tau']
    setting['horizon_len'] = horizon_len
    setting['n_action'] = env.action_space
    setting['ftr_size'] = ftr_transform.dimension

    trajectory = [
        Trajectory(ftr_transform.dimension, device, horizon_len)
            for _ in range(env.action_space)
    ]
    model = Model(device, setting)
    return {
            'env': env,
            'ftr_transform': ftr_transform,
            'trajectory': trajectory,
            'model': model,
            'device': device
            }

def get_device(setting):
    return torch.device('cuda'
            if torch.cuda.is_available() and not setting['cpu']
            else 'cpu')

def print_info(setting):
    print('---------------------')
    print('Pytorch version')
    print('Environment:', setting['env'])
    print('Algorithm:', 'Value iteration' if setting['algo'] == 'val' else 'Policy iteration')
    print('observation_space:', env.observation_space,
            'action_space:', env.action_space,
            'feature dimension:', ftr_transform.dimension)
    print('---------------------')
