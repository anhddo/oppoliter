import numpy as np
import numpy.random as npr

from ..dp_solver import DPSolver
from ..model import Agent


class OnlineValueIteration(object):
    def __init__(self):
        self.name = "Value iteration"

    def run(self, c, setting, env):
        n_state, n_action, n_step, p, alpha, n_episode = (
            setting["n_state"],
            setting["n_action"],
            setting["n_step"],
            setting["p"],
            setting["alpha"],
            setting["n_episode"],
        )

        yota = np.log(n_state * n_action * n_episode * n_step / p)
        bonus_constant = n_step ** 3 * yota
        env = env
        agent = Agent(setting)

        for h in range(n_step):
            for s in range(n_state):
                for a in range(n_action):
                    if np.sum(env.P[h, s, a, :]) > 0:
                        agent.Q[h, s, a] = n_step

        dp_solver = DPSolver()

        setting = setting
        n_state, n_action, n_step, p, alpha, n_episode = (
            setting["n_state"],
            setting["n_action"],
            setting["n_step"],
            setting["p"],
            setting["alpha"],
            setting["n_episode"],
        )
        v_optimal = dp_solver.optimal_value(env)
        bonus_constant = bonus_constant

        regret = np.zeros(n_episode)
        for episode in range(n_episode):
            start_state = np.random.choice(setting['state_per_stage'], size=1)[0]
            state = start_state
            for step in range(n_step):
                action = agent.greedy(step, state)
                agent.N[step, state, action] += 1
                next_state, reward = env.step(step, state, action)
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

            v_policy = dp_solver.policy_value(env, agent)
            regret[episode] = v_optimal[0, start_state] - v_policy[0, start_state]
        return regret
