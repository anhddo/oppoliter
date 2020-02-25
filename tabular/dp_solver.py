import numpy as np


class DPSolver:
    def __init__(self):
        pass

    def optimal_value(self, env):
        env_setting = env.env_setting
        n_state, n_action, n_step = (
            env_setting["n_state"],
            env_setting["n_action"],
            env_setting["n_step"],
        )

        state_value = np.zeros((n_step + 1, n_state))
        for step in reversed(range(0, n_step)):
            sa = []
            for state in range(n_state):
                q_value = np.zeros(n_action)
                for action in range(n_action):
                    next_state_transition = env.P[step, state, action, :]
                    v_next = state_value[step + 1, :]
                    expect_next_state_value = next_state_transition.dot(v_next)
                    reward = env.R[step, state, action]
                    q_value[action] = reward + expect_next_state_value
                state_value[step, state] = np.max(q_value)

        return state_value

    def policy_value(self, env, agent):
        env_setting = env.env_setting
        n_state, n_action, n_step = (
            env_setting["n_state"],
            env_setting["n_action"],
            env_setting["n_step"],
        )
        state_value = np.zeros((n_step + 1, n_state))
        pi = agent.policy_greedy()
        for step in reversed(range(n_step)):
            for state in range(n_state):
                action = pi[step, state]
                reward = env.R[step, state, action]
                next_state_transition = env.P[step, state, action, :]
                v_next = state_value[step + 1, :]
                state_value[step, state] = reward + next_state_transition.dot(v_next)
        return state_value
