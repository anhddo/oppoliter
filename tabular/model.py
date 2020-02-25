import numpy as np


class Agent(object):
    def __init__(self, env_setting):
        n_state, n_action, n_step = (
            env_setting["n_state"],
            env_setting["n_action"],
            env_setting["n_step"],
        )
        self.env_setting = env_setting
        self.Q = np.ones((n_step + 1, n_state, n_action)) * n_step
        self.N = np.zeros((n_step + 1, n_state, n_action)) + 1e-3
        self.V = np.zeros((n_step + 1, n_state))
        self.R_hat = np.zeros((n_step + 1, n_state, n_action))
        self.P_hat = np.zeros((n_step + 1, n_state, n_action, n_state))
        self.transition_count = np.zeros(
            (n_step + 1, n_state, n_action, n_state), dtype=np.int
        )
        self.Q[n_step, ...] = 0

    def greedy(self, step, state):
        return np.argmax(self.Q[step, state, :])

    def policy_greedy(self):
        setting = self.env_setting
        n_state, n_action, n_step, p = (
            setting["n_state"],
            setting["n_action"],
            setting["n_step"],
            setting["p"],
        )
        pi = np.zeros((n_step + 1, n_state), dtype=np.int)
        for step in range(n_step):
            for state in range(n_state):
                pi[step, state] = self.greedy(step, state)
        return pi

    def estimate_reward(self, step, state, action, reward):
        N = self.N[step, state, action]
        self.R_hat[step, state, action] = (
                                                  self.R_hat[step, state, action] * N + reward
                                          ) / (N + 1)
        self.N[step, state, action] = N + 1

    def estimate_transition(self, step, state, action, next_state):
        self.transition_count[step, state, action, next_state] += 1
        self.P_hat[step, state, action, :] = self.transition_count[
                                             step, state, action, :
                                             ] / np.sum(self.transition_count[step, state, action, :])

    def estimate_dynamic(self, step, state, action, reward, next_state):
        self.estimate_reward(step, state, action, reward)
        self.estimate_transition(step, state, action, next_state)


class Env(object):
    def __init__(self, **kargs):
        self.n_state = kargs["n_state"]
        self.n_step = kargs["n_step"]
        self.n_action = kargs["n_action"]
        self.stage_state = np.zeros((self.n_step, self.n_state), dtype=np.int)
        self.P = np.zeros((self.n_step, self.n_state, self.n_action, self.n_state))
        self.R = np.zeros((self.n_step, self.n_state, self.n_action))
        self.env_setting = kargs

    def step(self, step, state, action):
        prob = self.P[step, state, action, :]
        reward = self.R[step, state, action]
        if step == self.n_step - 1:
            next_state = -1
        else:
            next_state = np.random.choice(self.n_state, size=1, p=prob)[0]
        return next_state, reward

    def __str__(self):
        str = ''
        for action in range(self.n_action):
            str += 'action:{:.4f}, reward:{:.4f}'.format(action, self.R[0, 0, action])
            str += '\n'
        return str
