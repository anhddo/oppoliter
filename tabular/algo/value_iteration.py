import copy

import numpy as np
import numpy.random as npr

from ..dp_solver import DPSolver
from ..model import Agent


class OnlineValueIteration(object):
    def __init__(self, setting, env):
        self.name = "Value iteration"
        n_state, n_action, n_step, p, alpha, n_episode = (
            setting["n_state"],
            setting["n_action"],
            setting["n_step"],
            setting["p"],
            setting["alpha"],
            setting["n_episode"],
        )
        self.setting = setting
        self.bonus_constant = n_step ** 3 * np.log(
            n_state * n_action * n_episode * n_step / p
        )
        self.env = env
        self.agent = Agent(setting)

        for h in range(n_step):
            for s in range(n_state):
                for a in range(n_action):
                    if np.sum(self.env.P[h, s, a, :]) > 0:
                        self.agent.Q[h, s, a] = n_step

        self.dp_solver = DPSolver()

    def run(self, c=1):
        setting = self.setting
        n_state, n_action, n_step, p, alpha, n_episode = (
            setting["n_state"],
            setting["n_action"],
            setting["n_step"],
            setting["p"],
            setting["alpha"],
            setting["n_episode"],
        )
        agent = copy.deepcopy(self.agent)
        v_optimal = self.dp_solver.optimal_value(self.env)
        bonus_constant = self.bonus_constant

        regret = np.zeros(n_episode)
        for episode in range(n_episode):
            start_state = np.random.choice(setting['state_per_stage'], size=1)[0]
            state = start_state
            for step in range(n_step):
                action = agent.greedy(step, state)
                agent.N[step, state, action] += 1
                next_state, reward = self.env.step(step, state, action)
                bonus = c * np.sqrt(bonus_constant / agent.N[step, state, action])
                v_next = 0
                if step == n_step - 1:
                    v_next = agent.V[step + 1, next_state]
                q_hat_next = reward + v_next + bonus
                q_hat = agent.Q[step, state, action]
                q_hat = (1 - alpha) * q_hat + alpha * q_hat_next
                agent.Q[step, state, action] = q_hat
                agent.V[step, state] = min(np.max(agent.Q[step, state, :]), n_step)
                state = next_state

            v_policy = self.dp_solver.policy_value(self.env, agent)
            regret[episode] = v_optimal[0, start_state] - v_policy[0, start_state]
        return regret
