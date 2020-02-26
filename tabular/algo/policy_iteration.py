import numpy as np
import numpy.random as npr

from ..dp_solver import DPSolver
from ..model import Agent


class Tracking(object):
    def __init__(self, n_episode, n_step, n_state, n_action):
        self.bonus = np.zeros((n_episode, n_step, n_state, n_action))
        self.Q = np.ones((n_episode, n_step, n_state, n_action)) * n_step
        self.N = np.zeros((n_episode, n_step, n_state, n_action))


class PolicyIteration(object):
    def __init__(self, using_previous_estimate=True):
        if using_previous_estimate:
            self.name = "Policy iteration using previous estimate"
        else:
            self.name = "Policy iteration"
        self.using_previous_etimate = using_previous_estimate

    def run(self, c, setting, env):
        n_state, n_action, n_step, p, n_episode = (
            setting["n_state"],
            setting["n_action"],
            setting["n_step"],
            setting["p"],
            setting["n_episode"],
        )
        yota = np.log(n_state * n_action * n_episode * n_step / p)
        bonus_constant = n_step ** 3 * yota
        setting = setting
        agent = Agent(setting)
        dp_solver = DPSolver()

        tracking = Tracking(n_episode, n_step, n_state, n_action)

        env = env
        v_optimal = dp_solver.optimal_value(env)
        q_prev_estimate = np.zeros(agent.Q.shape)
        regret = np.zeros(n_episode)
        for episode in range(n_episode):
            pi = agent.policy_greedy()
            start_state = np.random.choice(setting['state_per_stage'], size=1)[0]
            state = start_state
            trajectory = []
            for step in range(n_step):
                action = pi[step, state]
                next_state, reward = env.step(step, state, action)
                trajectory.append((state, action, reward, next_state))
                state = next_state

            for step in range(n_step):
                state, action, reward, next_state = trajectory[step]
                agent.estimate_dynamic(step, state, action, reward, next_state)

            for step in reversed(range(n_step)):
                state, action, reward, next_state = trajectory[step]
                reward = agent.R_hat[step, state, action]
                p_hat = agent.P_hat[step, state, action, :]
                v_next_state = np.zeros(n_state)
                q_estimate = agent.Q
                if self.using_previous_etimate:
                    q_estimate = q_prev_estimate
                if step < n_step:
                    v_next_state = [
                        q_estimate[step + 1, state, pi[step + 1, state]] for state in range(n_state)
                    ]
                bonus = c * np.sqrt(bonus_constant * 1.0 / agent.N[step, state, action])
                q_hat = reward + np.dot(p_hat, v_next_state) + bonus

                agent.Q[step, state, action] = min(n_step, q_hat)
                #### tracking
                tracking.N[episode, step, state, action] = agent.N[step, state, action]
                tracking.Q[episode, step, state, action] = agent.Q[step, state, action]
                tracking.bonus[episode, step, state, action] = bonus
                #### tracking

            v_policy = dp_solver.policy_value(env, agent)
            regret[episode] = v_optimal[0, start_state] - v_policy[0, start_state]

        return regret
