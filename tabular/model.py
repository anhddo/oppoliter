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
        self.N = np.zeros((n_step + 1, n_state, n_action)) + 1e-10
        self.V = np.zeros((n_step + 1, n_state))
        self.R_hat = np.zeros((n_step + 1, n_state, n_action))
        self.P_hat = np.zeros((n_step + 1, n_state, n_action, n_state))
        self.transition_count = np.zeros(
            (n_step + 1, n_state, n_action, n_state), dtype=np.int
        )
        self.Q[n_step, ...] = 0
        self.pi = np.zeros((n_step + 1, n_state), dtype=np.int)

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
        self.pi[...] = 0
        for step in range(n_step):
            for state in range(n_state):
                self.pi[step, state] = self.greedy(step, state)
        return self.pi

    def estimate_reward(self, step, state, action, reward):
        N = self.N[step, state, action]
        self.R_hat[step, state, action] = (self.R_hat[step, state, action] * N + reward) / (N + 1)

    def estimate_transition(self, step, state, action, next_state):
        self.transition_count[step, state, action, next_state] += 1
        state_action_count = self.transition_count[step, state, action, :]
        self.P_hat[step, state, action, :] = state_action_count / np.sum(state_action_count)

    def estimate_dynamic(self, step, state, action, reward, next_state):
        self.estimate_reward(step, state, action, reward)
        self.estimate_transition(step, state, action, next_state)
