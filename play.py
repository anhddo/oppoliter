import argparse
from os import path

import matplotlib as mpl

mpl.use("Agg")
import gym
from linear_numpy.lm import Model
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--env", default='CartPole-v0')
    parser.add_argument("--path")
    parser.add_argument("--fourier-order", type=int, default=4)
    args = parser.parse_args()
    setting = vars(args)
    env = gym.make(setting['env'])
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
        print(action)
        next_state, reward, terminal, info = env.step(action)
        episode_reward += reward
        next_state = ftr_transform.transform(next_state)
        state = next_state


