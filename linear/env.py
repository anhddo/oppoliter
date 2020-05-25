import gym
import numpy as np
from numpy import cos
import gym_cartpole_swingup

class EnvWrapper:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.observation_space = self.env.observation_space.shape[0]
        if env_name == 'BipedalWalker-v3':
            self.action_space = self.env.action_space.shape[0]
        elif env_name == 'CartPoleSwingUp-v0':
            self.action_space = 2
        else:
            self.action_space = self.env.action_space.n
        self.tracking_value = 0
        self.reset_tracking_value = 0
        self._max_episode_steps = self.env._max_episode_steps


        self.min_clamp = 0
        self.max_clamp = self.env._max_episode_steps
        if self.env_name == 'CartPole-v0' or self.env_name == 'CartPole-v1':
            pass

        elif self.env_name == 'MountainCar-v0':
            self.max_clamp = 0
            self.min_clamp = -self.env._max_episode_steps
            self.reset_tracking_value = self.env._max_episode_steps

        elif self.env_name == 'Acrobot-v1':
            self.max_clamp = 100
            self.reset_tracking_value = 500

        elif self.env_name == 'LunarLander-v2':
            self.min_clamp = -500
            self.max_clamp = 200
            self.reset_tracking_value = 0

        self.tracking_value = self.reset_tracking_value


    def step(self, action):
        if self.env_name == 'CartPoleSwingUp-v0':
            if action == 0:
                action = -1
        state, true_reward, terminal, info = self.env.step(action)
        self.t += 1
        true_reward = float(true_reward)
        modified_reward = true_reward

        if self.env_name == 'CartPole-v0' or self.env_name == 'CartPole-v1':
            if terminal:
                modified_reward = 0
            self.tracking_value += true_reward
        elif self.env_name == 'CartPoleSwingUp-v0':
            self.tracking_value += true_reward

        elif self.env_name == 'MountainCar-v0':
            self.tracking_value = self.t
            if terminal:
                modified_reward = 0

        elif self.env_name == 'Acrobot-v1':
            self.tracking_value = self.t
        elif self.env_name == 'LunarLander-v2':
            self.tracking_value += true_reward


        return state, true_reward, modified_reward, terminal, info

    def reset(self):
        self.t = 0
        self.tracking_value = self.reset_tracking_value
        return self.env.reset()

