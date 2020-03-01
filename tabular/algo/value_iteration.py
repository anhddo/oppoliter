import numpy as np
import numpy.random as npr

from ..dp_solver import DPSolver
from ..model import Agent


class OnlineValueIteration(object):
    def __init__(self):
        self.name = "Chi Jin, et al 2018"

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

        dp_solver = DPSolver()
        v_optimal = dp_solver.optimal_value(setting, env)

        regret = np.zeros(n_episode)
        for episode in range(n_episode):
            start_state = np.random.choice(setting["state_per_stage"], size=1)[0]
            state = start_state
            for step in range(n_step):
                action = agent.greedy(step, state)
                agent.N[step, state, action] += 1
                next_state, reward = env.step(step, state, action)
                bonus = c * np.sqrt(bonus_constant / agent.N[step, state, action])
                v_next = agent.V[step + 1, next_state]
                q_hat_next = reward + v_next + bonus
                q_hat = agent.Q[step, state, action]
                agent.Q[step, state, action] = (1 - alpha) * q_hat + alpha * q_hat_next
                agent.V[step, state] = min(np.max(agent.Q[step, state, :]), n_step)
                state = next_state

            v_policy = dp_solver.policy_value(setting, env, agent)
            regret[episode] = v_optimal[0, start_state] - v_policy[0, start_state]
        return regret
