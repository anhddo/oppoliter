import numpy as np
import numpy.random as npr
import torch
from linear.lm import Model
import copy
from tqdm import tqdm
from scipy.special import softmax

VAL = 0
POL = 1
EPSILON_GREEDY = 2


class Politex:
    def __init__(:
        name = "Politex"


    def train( train_index, save_dir):
        init = initialize(setting)
        env, trajectory, model, ftr_transform, device =\
                init['env'], init['trajectory'], init['model'],\
                init['ftr_transform'], init['device']
        sum_modified_reward = 0
        terminal = True
        target_track, time_step = [], []

        state = None
        episode_count = 0
        env.reset()
        t = -1

        pbar = tqdm(total=setting['horizon_len'], leave=True)
        for i in range(T):
            for _ in range(tau):
                t += 1
                pbar.update()
                if terminal:
                    time_step.append(t)
                    target_track.append(env.tracking_value)
                    sum_modified_reward = 0
                    state = env.reset()
                    state = ftr_transform.transform(state)
                    episode_count += 1

                q = model.predict(state, setting['bonus']).squeeze(0).cpu().numpy()
                d = softmax(setting['lr'] * q)
                action = np.random.choice(env.action_space, 1, p=d)[0]
                model.action_count[action]+=1

                model.action_model[action].update_cov(state)
                next_state, true_reward, modified_reward, terminal, info = env.step(action)
                sum_modified_reward += modified_reward
                next_state = ftr_transform.transform(next_state)
                trajectory_per_action[action].append(state, modified_reward, next_state, terminal)
                state = next_state

            for e, m in zip(expert.action_model, model.action_model):
                e.reset_zeros()
                e.inv_cov = m.inv_cov
            policy = next_state_policy()
            for _ in range(n_eval):
                expert.update(
                        trajectory_per_action,
                        env,
                        discount,
                        device,
                        setting['bonus'],
                        policy
                    )
            policy = []
            for trajectory_per_action in trajectory:
                _, _, next_state, _ = trajectory_per_action.get_past_data()
                if next_state.shape[0] == 0:
                    policy.append(None)
                else:
                    policy.append(model.choose_action(next_state, setting['bonus']))

            with open(path.join(save_dir, 'result{}.pkl'.format(i)), 'wb') as f:
                pickle.dump([reward_track, time_step], f)

            for e, m in zip(expert.action_model, model.action_model):
                m.w += e.w

        env.reset()
        return target_track, time_step
