import torch
import gym
from gym.wrappers import Monitor
import argparse
from tqdm import trange
import numpy.random as npr
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from datetime import datetime
from PIL import Image

class ReplayBuffer():
    def __init__(self, N=10000):
        self.N = N
        self.buff = [[]] * N
        self.index = -1
        self.buffer_size = -1

    def append(self, s, a, r, s1, t):
        self.index = (self.index + 1) % self.N
        self.buffer_size = min(self.buffer_size + 1, self.N)
        self.buff[self.index] = (s, a, r, s1, t)

    def getbatch(self, batch_size):
        index = npr.choice(self.buffer_size, batch_size)
        batch = [self.buff[i] for i in index]
        s, a, r, s1, t = zip(*batch)
        s = torch.stack(s)
        s1 = torch.stack(s1)
        a = torch.stack(a).view(-1, 1)
        r = torch.stack(r).view(-1, 1)
        t = torch.stack(t).view(-1, 1)
        return s, a, r, s1, t

class DQN(torch.nn.Module):
    def __init__(self, n_observation, n_action):
        super(DQN, self).__init__()
        self.ln1 = torch.nn.Linear(n_observation, 50)
        self.ln2= torch.nn.Linear(50, n_action)
        torch.nn.init.kaiming_uniform_(self.ln1.weight)
        torch.nn.init.kaiming_uniform_(self.ln2.weight)

    def forward(self, s):
        s = F.relu(self.ln1(s))
        #s = F.relu(self.ln2(s))
        s = self.ln2(s)
        return s

    def take_action(self, s):
        with torch.no_grad():
            Q = policy_net(s)
            a = torch.argmax(Q).item()
            return a

def optimize_model(policy_net, target_net, buff, optimizer, device):
    if buff.buffer_size < setting['batch_size']:
        return
    s, a, r, s1, done = buff.getbatch(setting['batch_size'])

    Q = policy_net(s)
    Q = Q.gather(1, a)

    Q1 = policy_net(s1)
    a1 = torch.argmax(Q1, dim=1, keepdim=True)

    Q1 = target_net(s1)
    V1 = Q1.gather(1, a1)
    Q_t = r + V1 * (1 - done)
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(Q, Q_t)
    loss.backward()
    optimizer.step()
    return loss

def record(setting, policy_net):
    env = gym.make(setting['env'])
    #env = Monitor(env, 'tmp/video', force=True)
    device = torch.device('cpu')
    s = env.reset()
    s = torch.tensor(s, device=device)
    frames = []
    while True:
        a = policy_net.take_action(s)
        im = env.render(mode='rgb_array')
        im = np.copy(im)
        im = np.rollaxis(im, 2, 0)
        im = torch.tensor(im, device=device)
        im = torch.clamp(im, min=0, max=255)
        frames.append(im)
        s1, r, done, _ = env.step(a)
        s1 = torch.tensor(s1, device=device)
        s = s1
        if done:
            env.close()
            break
    frames = torch.stack(frames)
    print(frames.shape)
    return frames
    #frame = torch.stack(frames)
    #return frames
    #with open('tmp/cartpole1.gif', 'wb') as f:
    #    im = Image.new('RGB', frames[0].size)
    #    im.save(f, save_all=True, append_images=frames, duration=100, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-update", type=int, default=100)
    parser.add_argument("--env", default='CartPole-v0')
    args = parser.parse_args()
    setting = vars(args)

    device = torch.device('cpu')

    torch.set_default_dtype(torch.double)
    env = gym.make(setting['env'])
    n_observation = env.observation_space.shape[0]
    n_action = env.action_space.n

    policy_net = DQN(n_observation, n_action).to(device)
    target_net = DQN(n_observation, n_action).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    #optimizer = torch.optim.SGD(policy_net.parameters(), lr=0.01)
    optimizer = torch.optim.RMSprop(policy_net.parameters())

    buff = ReplayBuffer(setting['buffer_size'])
    s = env.reset()
    s = torch.tensor(s, requires_grad=False, device=device)

    episode_reward = 0
    episode = -1 
    epsilon = 1

    writer = SummaryWriter(log_dir='logs/{}-{}'.format(setting['env'], str( datetime.now())))

    frames = record(setting, policy_net)
    writer.add_images('drl/img', frames, 0)

    for step in trange(setting['step']):
        epsilon = max(epsilon * 0.999, .05)
        if npr.uniform() < epsilon:
            a = env.action_space.sample()
        else:
            a = policy_net.take_action(s)
        s1, r, done, _ = env.step(a)
        episode_reward += r

        s1 = torch.tensor(s1, requires_grad=False, device=device)
        r = torch.tensor(r, requires_grad=False, device=device)
        a = torch.tensor(a, requires_grad=False, device=device)
        done = torch.tensor(int(done), requires_grad=False, device=device)

        buff.append(s, a, r, s1, done)
        s = s1
        loss = optimize_model(policy_net, target_net, buff, optimizer, device)
        writer.add_scalar('drl/epsilon', epsilon, step)
        writer.add_scalar('drl/loss', loss if loss else 0, step)
        if done:
            episode += 1
            s = env.reset()
            s = torch.tensor(s).to(device)
            writer.add_scalar('drl/reward', episode_reward, step)
            episode_reward = 0
        if step % setting['target_update'] == 0:
            target_net.load_state_dict(policy_net.state_dict())

