import numpy as np
import numpy.random as npr


class FiniteMDP:
    def __init__(self, setting):
        n_step = setting["n_step"]
        n_action = setting["n_action"]
        state_per_stage = setting["state_per_stage"]

        n_state = n_step * state_per_stage
        setting["n_state"] = n_state
        self.stage_state = np.zeros((n_step, n_state), dtype=np.int)
        self.P = np.zeros((n_step, n_state, n_action, n_state))
        self.mean_reward = np.zeros((n_step, n_state, n_action))
        for step in range(n_step):
            current_states = range(step * state_per_stage, (step + 1) * state_per_stage)
            for state in current_states:
                for action in range(n_action):
                    self.mean_reward[step, state, action] = npr.rand()
                    is_last_step = step == n_step - 1
                    if is_last_step:
                        continue
                    next_states = range((step + 1) * state_per_stage, (step + 2) * state_per_stage)
                    prob_to_next_state = np.array(
                        [npr.randint(1, n_state) if i in next_states else 0 for i in range(n_state)])
                    prob_to_next_state = prob_to_next_state / sum(prob_to_next_state)
                    self.P[step, state, action, :] = prob_to_next_state
        self.random_reward = setting["random_reward"]
        self.n_step = n_step
        self.n_state = n_state

    def step(self, step, state, action):
        prob = self.P[step, state, action, :]
        reward = self.reward(step, state, action)
        is_last_step = step == self.n_step - 1
        next_state = -1 if is_last_step else np.random.choice(self.n_state, size=1, p=prob)[0]
        return next_state, reward

    def reward_deterministic(self, step, state, action):
        return self.mean_reward[step, state, action]

    def reward_random(self, step, state, action):
        return np.random.binomial(1, self.mean_reward[step, state, action])

    def reward(self, step, state, action):
        if self.random_reward:
            return self.reward_random(step, state, action)
        return self.mean_reward[step, state, action]

    def __str__(self):
        s = ""
        for action in range(self.n_action):
            s += "action:{:.4f}, reward:{:.4f}\n".format(action, self.mean_reward[0, 0, action])
        return s
