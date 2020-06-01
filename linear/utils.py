from .trajectory import Trajectory
from .fourier_transform import FourierTransform
from .lm import Model
from .env import EnvWrapper
from .ddqn import DQN
import torch
import pickle

def initialize(setting):
    env = EnvWrapper(setting['env'])
    device = get_device(setting)
    horizon_len = setting['step']
    setting['n_action'] = env.action_space
    setting['n_observation'] = env.observation_space
    if setting['use_nn']:
        ftr_transform = DQN(setting['n_observation'], env.action_space, 16, device).to(device)
        ftr_transform.load_state_dict(torch.load('tmp/{}-nn-model'.format(setting['env'])))
        ftr_transform.eval()
        ftr_transform.dimension = 16
    else:
        ftr_transform = FourierTransform(setting)
    setting['feature_size'] = ftr_transform.dimension

    #trajectory = [
    #    Trajectory(ftr_transform.dimension, device, horizon_len)
    #        for _ in range(env.action_space)
    #]
    model = Model(setting, device)
    return {
            'env': env,
            'ftr_transform': ftr_transform,
            #'trajectory': trajectory,
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
    print('observation_space:', setting['n_observation'],
            'action_space:', setting['n_action'],
            'feature dimension:', setting['feature_size'])
    print('---------------------')
