import numpy as np
import numpy.random as npr

from .model import Env


class FiniteMDP:
    def __init__(self, setting):
        n_action, state_per_stage, n_step = (
            setting["n_action"],
            setting["state_per_stage"],
            setting["n_step"],
        )
        n_state = n_step * state_per_stage
        env = Env(n_step=n_step, n_action=n_action, n_state=n_state)
        for step in range(n_step):
            current_states = range(step * state_per_stage, (step + 1) * state_per_stage)
            for state in current_states:
                for action in range(n_action):
                    if step < n_step - 1:
                        next_states = range(
                            (step + 1) * state_per_stage, (step + 2) * state_per_stage
                        )
                        prob_to_next_state = np.array(
                            [
                                npr.randint(1, n_state) if i in next_states else 0
                                for i in range(n_state)
                            ]
                        )
                        prob_to_next_state = prob_to_next_state / sum(
                            prob_to_next_state
                        )
                        env.P[step, state, action, :] = prob_to_next_state
                    env.R[step, state, action] = npr.rand()

        self.env = env
        setting["n_state"] = n_state
