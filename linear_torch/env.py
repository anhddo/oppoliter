import gym
import numpy as np

class EnvWrapper:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.observation_space = self.env.observation_space.shape[0]
        if env_name == 'BipedalWalker-v3':
            self.action_space = self.env.action_space.shape[0]
        else:
            self.action_space = self.env.action_space.n
        self._max_episode_steps = self.env._max_episode_steps
        self.max_clamp = self.env._max_episode_steps

        self.min_clamp = 0
        if self.env_name == 'CartPole-v0' or self.env_name == 'CartPole-v1':
            self.max_clamp = self.env._max_episode_steps

        elif self.env_name == 'MountainCar-v0':
            self.max_clamp = 1e5

        elif self.env_name == 'Acrobot-v1':
            self.max_clamp = 1e4

        elif self.env_name == 'LunarLander-v2':
            self.min_clamp = -500
            self.max_clamp = 200

    def step(self, action):
        state, true_reward, terminal, info = self.env.step(action)
        self.t += 1
        if self.env_name == 'CartPole-v0' or self.env_name == 'CartPole-v1':
            modified_reward = true_reward
            if terminal:
                modified_reward = self.t - self.env._max_episode_steps

        elif self.env_name == 'MountainCar-v0':
            #modified_reward = true_reward
            if terminal and self.t < self.env._max_episode_steps:
                modified_reward = self.max_clamp
            else:
                modified_reward = 0
                #modified_reward = np.exp(2*np.abs(state[0] + 0.6))

        elif self.env_name == 'Acrobot-v1':
            modified_reward = 0
            if terminal:
                if self.t < self.env._max_episode_steps:
                    modified_reward = self.max_clamp
        else:
            modified_reward = true_reward


        return state, true_reward, modified_reward, terminal, info

    def reset(self):
        self.t = 0
        return self.env.reset()

