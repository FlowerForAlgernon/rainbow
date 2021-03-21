import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, c, h, w, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)

        self.fc4 = nn.Linear(convw*convh*64, 512)
        self.fc5 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


class DuelingDQN(nn.Module):
    def __init__(self, c, h, w, n_actions):
        super(Dueling_DQN, self).__init__()
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)

        self.advantage1 = nn.Linear(convw*convh*64, 512)
        self.advantage2 = nn.Linear(512, n_actions)

        self.value1 = nn.Linear(convw*convh*64, 512)
        self.value2 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        advantage = F.relu(self.advantage1(x))
        advantage = self.advantage2(advantage)
        value = F.relu(self.value1(x))
        value = self.value2(value)
        return value + advantage - advantage.mean()


class NoisyLinear(nn.Linear):
    def __init__(self, in_size, out_size):
        super(NoisyLinear, self).__init__(in_size, out_size)
        self.in_size = in_size
        self.out_size = out_size
        self.mu = 1.0 / np.sqrt(in_size)
        self.sigma = 0.5 / np.sqrt(in_size)

        torch.nn.init.uniform_(self.weight, a=-self.mu, b=self.mu)
        torch.nn.init.uniform_(self.bias, a=-self.mu, b=self.mu)

        self.sigma_w = nn.Parameter(torch.full((out_size, in_size), self.sigma), True)
        self.sigma_b = nn.Parameter(torch.full((out_size,), self.sigma), True)
        self.register_buffer("noise_in", torch.zeros(in_size))
        self.register_buffer("noise_out", torch.zeros(out_size))
        self.register_buffer("epsilon_w", torch.zeros(out_size, in_size))
        self.register_buffer("epsilon_b", torch.zeros(out_size))
        self.reset_noise()

    def forward(self, x: torch.Tensor):
        w = self.weight + self.sigma_w * self.epsilon_w
        b = self.bias + self.sigma_b * self.epsilon_b
        return F.linear(x, w, b)

    def reset_noise(self):
        self.noise_in.normal_()
        self.noise_out.normal_()
        noise_w = torch.ger(self.noise_out, self.noise_in)
        noise_b = self.noise_out
        self.epsilon_w = torch.sign(noise_w) * torch.sqrt(torch.abs(noise_w))
        self.epsilon_b = torch.sign(noise_b) * torch.sqrt(torch.abs(noise_b))


class NoisyDQN(nn.Module):
    def __init__(self, c, h, w, n_actions):
        super(NoisyDQN, self).__init__()
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)

        self.advantage1 = NoisyLinear(convw*convh*64, 512)
        self.advantage2 = NoisyLinear(512, n_actions)

        self.value1 = NoisyLinear(convw*convh*64, 512)
        self.value2 = NoisyLinear(512, 1)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        advantage = F.relu(self.advantage1(x))
        advantage = self.advantage2(advantage)
        value = F.relu(self.value1(x))
        value = self.value2(value)
        return value + advantage - advantage.mean()

    def reset_noise(self):
        self.advantage1.reset_noise()
        self.advantage2.reset_noise()
        self.value1.reset_noise()
        self.value2.reset_noise()


class DistributionalDQN(nn.Module):
    def __init__(self, c, h, w, n_actions, atom_size, support):
        super(DistributionalDQN, self).__init__()

        self.support = support
        self.n_actions = n_actions
        self.atom_size = atom_size

        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)

        self.advantage1 = NoisyLinear(convw*convh*64, 512)
        self.advantage2 = NoisyLinear(512, n_actions*atom_size)

        self.value1 = NoisyLinear(convw*convh*64, 512)
        self.value2 = NoisyLinear(512, atom_size)

    def forward(self, x):
        dist = self.get_dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def get_dist(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        advantage = F.relu(self.advantage1(x))
        advantage = self.advantage2(advantage)
        value = F.relu(self.value1(x))
        value = self.value2(value)

        advantage = advantage.view(-1, self.n_actions, self.atom_size)
        value = value.view(-1, 1, self.atom_size)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)

        return dist

    def reset_noise(self):
        self.advantage1.reset_noise()
        self.advantage2.reset_noise()
        self.value1.reset_noise()
        self.value2.reset_noise()