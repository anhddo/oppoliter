import tensorflow as tf

class Trajectory:
    def __init__(self, setting, device, horizon_len=10000):
        self.index = -1
        D = setting['feature_size']
        with tf.device("/gpu:0"):
            self.state = tf.zeros((horizon_len, D), dtype=tf.dtypes.double)
            self.next_state = tf.zeros((horizon_len, D), dtype=tf.dtypes.double)
            self.reward = tf.zeros(horizon_len, dtype=tf.dtypes.double)
            self.terminal = tf.zeros(horizon_len, dtype=tf.dtypes.double)
            self.horizon_len = horizon_len
            self.last_index = -1

            self.inv_cov = 10 * tf.eye(setting['feature_size'], dtype=tf.dtypes.double)
            self.last_inv_cov = 10 * tf.eye(setting['feature_size'], dtype=tf.dtypes.double)


    def append(self, state, reward, next_state, terminal):
        self.last_index = min(self.last_index + 1, self.horizon_len)
        self.index = (self.index + 1) % self.horizon_len
        old_state = self.state[self.index, :]

        self.update_cov(old_state, remove=True)

        self.state[self.index, :] = state
        self.next_state[self.index, :] = next_state
        self.reward[self.index] = reward
        self.terminal[self.index] = int(terminal)

        self.update_cov(state, remove=False)

    def update_cov(self, state_t, remove=False):
        state_t = state_t.view(1, -1).cpu()
        A = self.last_inv_cov
        d = 1. + state_t.mm(A).mm(state_t.T)
        U = A.mm(state_t.T).mm(state_t.mm(A)) / d
        if remove:
            self.last_inv_cov += U
        else:
            self.last_inv_cov -= U


    def get_past_data(self):
        index = self.last_index + 1
        return (self.state[:index, :],
                self.reward[:index],
                self.next_state[:index, :],
                self.terminal[:index])

    def reset(self):
        self.index = -1
