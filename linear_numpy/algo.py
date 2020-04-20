import matplotlib as mpl

mpl.use("Agg")
import numpy as np
from numpy.linalg import inv


class AverageReward:
    def __init__(self, env, model, ftr_transform, trajectory_per_action, setting):
        self.name = "Least square value iteration"
        self.regulization_matrix = 1e-4 * np.eye(ftr_transform.dimension)
        self.env = env
        self.ftr_transform = ftr_transform
        self.trajectory_per_action = trajectory_per_action
        self.total_step = setting['step']
        self.discount = setting['discount']
        self.model = model

    def train(self):
        episode_reward = 0
        sum_modified_reward = 0
        rewards = [0] * 10
        terminal = True
        reward_track, time_step = [], []

        state=None
        for t in range(self.total_step):
            if terminal:
                state = self.env.reset()
                state = self.ftr_transform.transform(state)
                rewards.append(episode_reward)
                print('===avg reward:',int(np.mean(rewards)), 
                        'true reward:', episode_reward,
                        'modified reward:', sum_modified_reward,
                        'step:',t, '===')
                reward_track.append(episode_reward)
                time_step.append(t)
                episode_reward, sum_modified_reward = 0, 0
                del rewards[0]
            action = self.model.choose_action(state)[0]
            next_state, true_reward, modified_reward, terminal, info = self.env.step(action)
            episode_reward += true_reward
            sum_modified_reward += modified_reward
            next_state = self.ftr_transform.transform(next_state)
            self.trajectory_per_action[action].append(state, modified_reward, next_state, terminal)
            state = next_state
            self.update_model()
        return reward_track, time_step

    def update_model(self):
        for least_square_model, trajectory in zip(self.model.action_model, self.trajectory_per_action):
            state, reward, next_state, terminal = trajectory.get_past_data()
            if state.shape[0] == 0:
                continue
            reward = reward.reshape(-1, 1)
            terminal = terminal.reshape(-1, 1)
            Q_next = self.model.predict(next_state)
            V_next = np.max(Q_next, axis=1).reshape(-1, 1)
            b = least_square_model.bonus(state)
            V_next = np.clip(V_next, 0, self.env._max_episode_steps)
            least_square_model.cov = state.T.dot(state) + self.regulization_matrix
            least_square_model.inv_cov = inv(least_square_model.cov)
            Q = (reward + self.discount * V_next) * (1 - terminal)
            least_square_model.w = least_square_model.inv_cov.dot(state.T.dot(Q))
            assert least_square_model.cov.shape == (self.model.D, self.model.D)
            assert least_square_model.inv_cov.shape == (self.model.D, self.model.D)
            assert least_square_model.w.shape == (self.model.D, 1)
            assert b.shape[1] == 1
            assert Q.shape[1] == 1

    def test(self):
        episode_reward = 0
        state = env.reset()
        state = ftr_transform.transform(state)
        while True:
            action = model.choose_action(state)[0]
            next_state, reward, terminal, info = env.step(action)
            episode_reward += reward
            next_state = ftr_transform.transform(next_state)
            state = next_state
            if terminal:
                return episode_reward


