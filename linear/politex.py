import numpy as np
import numpy.random as npr
import torch
from linear.lm import Model
import copy
from tqdm import tqdm
#from scipy.special import softmax
from .utils import initialize, print_info
from .lm import Model
from os import path
import pickle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.nn.functional import mse_loss, softmax



class Politex:
    def __init__(self):
        name = "Politex"

    def choose_action(self, model, state, action_space, lr, exploration_bonus, use_bonus, n):
        q = model.Q(state, False)#.squeeze(0)
        if use_bonus:
            pass
        p = softmax(lr * q, dim=1)
        A = torch.multinomial(p, n)
        return A

    def train(self, train_index, setting):
        init = initialize(setting)
        writer = SummaryWriter(log_dir='logs/{}-politex{}'\
                .format(setting['env'], '-optimistic' if setting['bonus'] else '') + str( datetime.now()))
        env, trajectory, model, ftr_transform, device =\
                init['env'], init['trajectory'], init['model'],\
                init['ftr_transform'], init['device']
        sum_modified_reward = 0
        terminal = False
        target_track, time_step = [], []
        state = env.reset()
        state = ftr_transform.transform(state)

        episode_count = 0
        t = -1
        expert = Model(setting, device)

        pbar = tqdm(total=setting['step'], leave=True)
        exploration_bonus = [[]]*setting['n_action']
        q_sum = 0
        setting['K'] = int(np.sqrt(setting['step']))
        setting['tau'] = int(np.sqrt(setting['step']))
        print('Phase', setting['K'], 'tau', setting['tau'])
        for i in range(setting['K']):
            #if setting['on_policy']:
            #    for e in trajectory:
            #        e.reset()
            #    for e in expert.action_model:
            #        pass
            #        e.reset_cov()
            for _ in range(setting['tau']):
                pbar.update()
                if terminal:
                    time_step.append(t)
                    target_track.append(env.tracking_value)
                    writer.add_scalar('ls/q', torch.max(expert.Q(state, setting['bonus'])), t)
                    writer.add_scalar('ls/reward', env.tracking_value, t)
                    writer.add_scalar('ls/t', env.t, t)
                    state = env.reset()
                    state = ftr_transform.transform(state)
                    episode_count += 1
                    sum_modified_reward = 0

                t += 1
                q = model.Q(state, False).squeeze(0)
                if setting['bonus']:
                    b = torch.zeros(setting['n_action'])
                    for j in range(setting['n_action']):
                        for inv_cov in exploration_bonus[j]:
                            b[j] += torch.sqrt(state.mm(inv_cov).mm(state.T)).item()
                    q += b
                #writer.add_scalar('politex/q_model', torch.max(q), t)
                #q_0 = torch.max(expert.Q(state, setting['bonus']))
                #q_sum += q_0
                #writer.add_scalar('politex/q_sum', q_sum, t)
                #writer.add_scalar('politex/lr', lr, t)
                lr = setting['lr']
                action = self.choose_action(model, state, env.action_space, lr, exploration_bonus, setting['bonus'], 1)
                next_state, true_reward, modified_reward, terminal, info = env.step(action.item())
                sum_modified_reward += modified_reward
                next_state = ftr_transform.transform(next_state)
                trajectory[action].append(state, modified_reward, next_state, terminal)
                expert.action_model[action].update_cov(state)
                state = next_state
                writer.add_scalar('ls/reward_raw', modified_reward, t)


            policy = []
            for trajectory_per_action in trajectory:
                _, _, next_state, _ = trajectory_per_action.get_past_data()
                if next_state.shape[0] == 0:
                    policy.append(None)
                else:

                    policy.append(self.choose_action(next_state, False))

            #print(policy)
            #import pdb; pdb.set_trace();
            n_eval = 0
            for e in expert.action_model:
                pass
                e.reset_w()
            #while True:
            #    n_eval += 1
            #    loss = 0
            loss = 0
            for _ in range(setting['n_eval']):
                w = [m.w for m in expert.action_model]
                expert.average_reward_algorithm(trajectory=trajectory, env=env,\
                        discount=setting['discount'], bonus=setting['bonus'], policy=policy)
                for o_w, m in zip(w, expert.action_model):
                    loss += mse_loss(o_w, m.w)
                #if loss < 1e-4:
            writer.add_scalar('politex/mse_w', loss, t)
            #writer.add_scalar('politex/n_eval', n_eval, t)
                    #break

            for e, m in zip(expert.action_model, model.action_model):
                m.w += e.w

            if setting['bonus']:
                for inv_cov_list, action_model in zip(exploration_bonus, expert.action_model):
                    inv_cov_list.append(action_model.inv_cov.clone().detach())

        pbar.close()
        env.reset()
        ftr_transform.save()
        with open(path.join(setting['save_dir'], 'result{}.pkl'.format(train_index)), 'wb') as f:
            pickle.dump([target_track, time_step], f)
