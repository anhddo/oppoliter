import gym
import numpy as np

class EnvWrapper:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self._max_episode_steps = self.env._max_episode_steps

    def step(self, action):
        state, true_reward, terminal, info = self.env.step(action)
        self.t += 1
        if self.env_name == 'CartPole-v0' or self.env_name == 'CartPole-v1':
            modified_reward = true_reward
            if terminal:
                modified_reward = self.t - self.env._max_episode_steps

        if self.env_name == 'MountainCar-v0':
            modified_reward = np.exp(state[0] + 0.6 ) * 1e3
            if terminal and self.t < self.env._max_episode_steps:
                modified_reward = 1e6

        return state, true_reward, modified_reward, terminal, info

    def reset(self):
        self.t = 0
        return self.env.reset()

