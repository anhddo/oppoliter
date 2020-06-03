import gym
import numpy as np
from numpy import cos
import gym_cartpole_swingup
import random
import numpy.random as npr

def env_step(env, action, t):
    state, reward, done, _ = env.step(action)
    if env.spec._env_name in ['MountainCar', 'Acrobot']:
        reward = 0
        if done and t < env._max_episode_steps:
            reward = 1

    if env.spec._env_name == 'Acrobot':
        state = env.state
    return state, reward, done

def env_reset(env):
    state = env.reset()
    if env.spec._env_name == 'Acrobot':
        state = env.state
    return state

#class EnvWrapper:
#    def __init__(self, setting):
#        env_name = setting['env']
#        self.random_reward = setting['random_reward']
#        self._env = gym.make(env_name)
#        self.env_name = env_name
#        self._env.seed(random.randint(0, 9999))
#        self.observation_space = self._env.observation_space.shape[0]
#        self.action_space = self._env.action_space.n
#        if env_name == 'BipedalWalker-v3':
#            self.action_space = self._env.action_space.shape[0]
#        elif env_name == 'CartPoleSwingUp-v0':
#            self.action_space = 2
#        elif env_name == 'Acrobot-v1':
#            self.observation_space = 4
#            pass
#        self.tracking_value = 0
#        self.reset_tracking_value = 0
#        self._max_episode_steps = self._env._max_episode_steps
#
#
#        self.min_clamp = 0
#        self.max_clamp = self._env._max_episode_steps
#        if self.env_name == 'CartPole-v0' or self.env_name == 'CartPole-v1':
#            pass
#
#        elif self.env_name == 'MountainCar-v0':
#            self.max_clamp = self._env._max_episode_steps
#            self.min_clamp = -self._env._max_episode_steps
#
#        elif self.env_name == 'Acrobot-v1':
#            self.max_clamp = self._env._max_episode_steps
#            self.min_clamp = -self._env._max_episode_steps
#            self.reset_tracking_value = 0
#
#
#        elif self.env_name == 'LunarLander-v2':
#            self.min_clamp = -500
#            self.max_clamp = 200
#            self.reset_tracking_value = 0
#
#        self.tracking_value = self.reset_tracking_value
#
#
#    def step(self, action):
#        if self.env_name == 'CartPoleSwingUp-v0':
#            if action == 0:
#                action = -1
#        state, true_reward, terminal, info = self._env.step(action)
#        self.t += 1
#        true_reward = float(true_reward)
#        modified_reward = true_reward
#
#        if self.env_name == 'CartPole-v0' or self.env_name == 'CartPole-v1':
#            #if terminal a:
#            if terminal and self.t < self._env._max_episode_steps:
#                #print(terminal, modified_reward, self.t)
#                modified_reward = 0
#            self.tracking_value += modified_reward
#        elif self.env_name == 'Acrobot-v1':
#            state = self._env.state
#            height = -cos(state[0]) - cos(state[1] + state[0])
#            if height > 1.9:
#                modified_reward = 0
#
#            self.tracking_value += modified_reward
#        elif self.env_name == 'CartPoleSwingUp-v0':
#            self.tracking_value += true_reward
#
#        elif self.env_name == 'MountainCar-v0':
#            self.tracking_value += true_reward
#            if terminal and self.t < self._env._max_episode_steps:
#                modified_reward = 0
#                if self.random_reward:
#                    if npr.uniform() < 0.1:
#                        modified_reward = 0
#                    else:
#                        modified_reward = -1
#        elif self.env_name == 'LunarLander-v2':
#            self.tracking_value += true_reward
#
#        return state, true_reward, modified_reward, terminal, info
#
#    def reset(self):
#        print('reset')
#        self.t = 0
#        self.tracking_value = self.reset_tracking_value
#        state = self._env.reset()
#        if self.env_name == 'Acrobot-v1':
#            pass
#            return self._env.state
#        return state
#
