import argparse
import os
import gym
import numpy as np
import random
from gym import wrappers
import torch
import torch.nn as nn
import gym_cartpole_swingup

def fanin_init(size, fanin=None):
    """
    weight initializer known from https://arxiv.org/abs/1502.01852
    :param size:
    :param fanin:
    :return:
    """
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h1=400, h2=300, eps=0.03):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1 + action_dim, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, 1)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        """
        return critic Q(s,a)
        :param state: state [n, state_dim] (n is batch_size)
        :param action: action [n, action_dim]
        :return: Q(s,a) [n, 1]
        """

        s1 = self.relu(self.fc1(state))
        x = torch.cat((s1, action), dim=1)

        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, h1=400, h2=300, eps=0.03):
        """
        :param state_dim: int
        :param action_dim: int
        :param action_lim: Used to limit action space in [-action_lim,action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, action_dim)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        """
        return actor policy function Pi(s)
        :param state: state [n, state_dim]
        :return: action [n, action_dim]
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc3(x)) # tanh limit (-1, 1)
        return action
class Config:
    def __init__(self):
        self.env = None
        self.episodes = None
        self.max_steps = None
        self.max_buff = None
        self.batch_size = None
        self.state_dim = None
        self.state_high = None
        self.state_low = None
        self.seed = None

        self.output = 'out'

        self.action_dim = None
        self.action_high = None
        self.action_low = None
        self.action_lim = None

        self.learning_rate = None
        self.learning_rate_actor = None

        self.gamma = None
        self.tau = None
        self.epsilon = None
        self.eps_decay = None
        self.epsilon_min = None

        self.use_cuda = True

        self.checkpoint = False
        self.checkpoint_interval = None

        self.use_matplotlib = False

        self.record = False
        self.record_ep_interval = None


class Trainer:
    def __init__(self, agent, env, config: Config, record=False):

        self.agent = agent
        self.config = config
        self.env = env
        self.env.seed(config.seed)
        self.agent.is_training = True


    def train(self, pre_episodes=0, pre_total_step=0):
        total_step = pre_total_step

        all_rewards = []
        for ep in range(pre_episodes + 1, self.config.episodes + 1):
            s0 = self.env.reset()
            self.agent.reset()

            done = False
            step = 0
            actor_loss, critics_loss, reward = 0, 0, 0

            # decay noise
            self.agent.decay_epsilon()

            while not done:
                #self.env.render()
                action = self.agent.get_action(s0)
                s1, r1, done, info = self.env.step(action)
                self.agent.buffer.add(s0, action, r1, done, s1)
                s0 = s1

                if self.agent.buffer.size() > self.config.batch_size:
                    loss_a, loss_c = self.agent.learning()
                    actor_loss += loss_a
                    critics_loss += loss_c

                reward += r1
                step += 1
                total_step += 1

                if step + 1 > self.config.max_steps:
                    break

            all_rewards.append(reward)
            avg_reward = float(np.mean(all_rewards[-100:]))

            print('total step: %5d, episodes %3d, episode_step: %5d, episode_reward: %5f' % (
                total_step, ep, step, reward))

class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)

def soft_update(target, source, tau=0.001):
    """
    update target by target = tau * source + (1 - tau) * target
    :param target: Target network
    :param source: source network
    :param tau: 0 < tau << 1
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    """
    update target by target = source
    :param target: Target network
    :param source: source network
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = []
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer = []
        self.count = 0


class DDPG:
    def __init__(self, config: Config):
        self.config = config
        self.init()

    def init(self):
        self.state_dim = self.config.state_dim
        self.action_dim = self.config.action_dim
        self.batch_size = self.config.batch_size
        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon
        self.is_training = True
        self.randomer = OUNoise(self.action_dim)
        self.buffer = ReplayBuffer(self.config.max_buff)

        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_target = Actor(self.state_dim, self.action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.config.learning_rate_actor)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.config.learning_rate)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        if self.config.use_cuda:
            self.cuda()

    def learning(self):
        s1, a1, r1, t1, s2 = self.buffer.sample_batch(self.batch_size)
        # bool -> int
        t1 = (t1 == False) * 1
        s1 = torch.tensor(s1, dtype=torch.float)
        a1 = torch.tensor(a1, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float)
        t1 = torch.tensor(t1, dtype=torch.float)
        s2 = torch.tensor(s2, dtype=torch.float)
        if self.config.use_cuda:
            s1 = s1.cuda()
            a1 = a1.cuda()
            r1 = r1.cuda()
            t1 = t1.cuda()
            s2 = s2.cuda()

        a2 = self.actor_target(s2).detach()
        target_q = self.critic_target(s2, a2).detach()
        y_expected = r1[:, None] + t1[:, None] * self.config.gamma * target_q
        y_predicted = self.critic.forward(s1, a1)

        # critic gradient
        critic_loss = nn.MSELoss()
        loss_critic = critic_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # actor gradient
        pred_a = self.actor.forward(s1)
        loss_actor = (-self.critic.forward(s1, pred_a)).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # Notice that we only have gradient updates for actor and critic, not target
        # actor_optimizer.step() and critic_optimizer.step()

        soft_update(self.actor_target, self.actor, self.config.tau)
        soft_update(self.critic_target, self.critic, self.config.tau)

        return loss_actor.item(), loss_critic.item()


    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def decay_epsilon(self):
        self.epsilon -= self.config.eps_decay

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        if self.config.use_cuda:
            state = state.cuda()

        action = self.actor(state).detach()
        action = action.squeeze(0).cpu().numpy()
        action += self.is_training * max(self.epsilon, self.config.epsilon_min) * self.randomer.noise()
        action = np.clip(action, -1.0, 1.0)

        self.action = action
        return action

    def reset(self):
        self.randomer.reset()



class OUNoise:
    """docstring for OUNoise"""
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

parser = argparse.ArgumentParser(description='')
parser.add_argument('--train', dest='train', action='store_true', help='train model')
parser.add_argument('--test', dest='test', action='store_true', help='test model')
parser.add_argument('--env', default='Pendulum-v0', type=str, help='gym environment')
parser.add_argument('--gamma', default=0.99, type=float, help='discount')
parser.add_argument('--episodes', default=200, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epsilon', default=1.0, type=float, help='noise epsilon')
parser.add_argument('--eps_decay', default=0.001, type=float, help='epsilon decay')
parser.add_argument('--max_buff', default=1000000, type=int, help='replay buff size')
parser.add_argument('--output', default='out', type=str, help='result output dir')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='use cuda')
parser.add_argument('--model_path', type=str, help='if test mode, import the model')
parser.add_argument('--load_config', type=str, help='load the config from obj file')

step_group = parser.add_argument_group('step')
step_group.add_argument('--customize_step', dest='customize_step', action='store_true', help='customize max step per episode')
step_group.add_argument('--max_steps', default=1000, type=int, help='max steps per episode')

record_group = parser.add_argument_group('record')
record_group.add_argument('--record', dest='record', action='store_true', help='record the video')
record_group.add_argument('--record_ep_interval', default=20, type=int, help='record episodes interval')

checkpoint_group = parser.add_argument_group('checkpoint')
checkpoint_group.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='use model checkpoint')
checkpoint_group.add_argument('--checkpoint_interval', default=500, type=int, help='checkpoint interval')

retrain_group = parser.add_argument_group('retrain')
retrain_group.add_argument('--retrain', dest='retrain', action='store_true', help='retrain model')
retrain_group.add_argument('--retrain_model', type=str, help='retrain model path')

args = parser.parse_args()
config = Config()
config.env = args.env
# config.env = 'MountainCarContinuous-v0'
# Pendulum-v0 # Walker2d-v2 # HalfCheetah-v1
config.gamma = args.gamma
config.episodes = args.episodes
config.max_steps = args.max_steps
config.batch_size = args.batch_size
config.epsilon = args.epsilon
config.eps_decay = args.eps_decay
config.max_buff = args.max_buff
config.output = args.output
config.use_cuda = args.cuda
config.checkpoint = args.checkpoint
config.checkpoint_interval = args.checkpoint_interval

config.learning_rate = 1e-3
config.learning_rate_actor = 1e-4
config.epsilon_min = 0.001
config.epsilon = 1.0
config.tau = 0.001

# env = gym.make() is limited by TimeLimit, there is a default max step.
# If you want to control the max step every episode, do env = gym.make(config.env).env
env = None
if args.customize_step:
    env = gym.make(config.env).env
else:
    env = gym.make(config.env)

env = NormalizedEnv(env)
config.action_dim = int(env.action_space.shape[0])
config.action_lim = float(env.action_space.high[0])
config.state_dim = int(env.observation_space.shape[0])

if args.load_config is not None:
        config = load_obj(args.load_config)

agent = DDPG(config)

trainer = Trainer(agent, env, config,
                  record=args.record)
trainer.train()
