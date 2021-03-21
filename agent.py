import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from memory import Transition, ReplayMemory, PrioritizedReplayMemory, NStepMemory
from DQN import DQN, DuelingDQN, NoisyDQN, DistributionalDQN

class Agent:
    def __init__(self, config):
        # Distributional DQN
        self.support = torch.linspace(config.v_min, config.v_max, config.atom_size).to(config.device)

        self.policy_net = DistributionalDQN(config.c, config.h, config.w, config.n_actions, config.atom_size, self.support).to(config.device)
        self.target_net = DistributionalDQN(config.c, config.h, config.w, config.n_actions, config.atom_size, self.support).to(config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        #self.memory = ReplayMemory(config.memory_size)
        self.memory = PrioritizedReplayMemory(config.memory_size, config.alpha)
        self.memory_n = NStepMemory(config.memory_size, config.alpha, config.gamma, config.n_step)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=config.learning_rate, eps=0.001, alpha=0.95)

    @staticmethod
    def get_state(obs, config):
        state = np.array(obs)[14:77,:,:]
        state = np.ascontiguousarray(state.transpose((2, 0, 1)), dtype=np.float)
        state = torch.from_numpy(state / 255)
        return state.unsqueeze(0).to(config.device)

    def select_action(self, state, epsilon, config):
        if random.random() > epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1,1).to(config.device)
        else:
            return torch.tensor([[random.randrange(4)]], device=config.device, dtype=torch.long)

    def transition_to_tensor(self, transitions):
        for i in range(len(transitions)):
            transitions[i][0] = torch.tensor(transitions[i][0]).to(config.device)
            transitions[i][1] = torch.tensor(transitions[i][1]).to(config.device)
            transitions[i][2] = torch.tensor(transitions[i][2]).to(config.device)
            transitions[i][3] = torch.tensor(transitions[i][3]).to(config.device)
        return transitions

    def optimize_model(self, config):
        #transitions = self.memory.sample(config.batch_size)
        # PrioritizedReplayMemory
        transitions, weights, indices = self.memory.sample(config.batch_size, config.beta)
        transitions = self.transition_to_tensor(transitions)
        batch = Transition(*zip(*transitions))
        loss, weights_loss = self.get_loss(batch, config, weights, config.gamma)

        # N Step
        transitions_n, _, _ = self.memory_n.sample_from_indices(config.batch_size, config.beta, indices)
        transitions_n = self.transition_to_tensor(transitions_n)
        batch_n = Transition(*zip(*transitions_n))
        gamma_n = config.gamma ** config.n_step
        loss_n, weights_loss_n = self.get_loss(batch_n, config, weights, gamma_n)
        weights_loss += weights_loss_n

        self.optimizer.zero_grad()
        #loss.backward()
        # PrioritizedReplayMemory
        weights_loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # PrioritizedReplayMemory
        loss_for_prior = loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + config.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        # N Step
        self.memory_n.update_priorities(indices, new_priorities)

        # Noisy Net
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def get_loss(self, batch, config, weights, gamma):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=config.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_action_values = torch.zeros(config.batch_size, device=config.device).unsqueeze(1)
        next_state_action_values[non_final_mask] = self.target_net(non_final_next_states).gather(
                1, self.policy_net(non_final_next_states).detach().argmax(dim=1, keepdim=True)
            ).detach()

        expected_state_action_values = reward_batch + gamma * next_state_action_values
        #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        # PrioritizedReplayMemory
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction="none")
        weights = torch.FloatTensor(np.array(weights).reshape(-1, 1)).to(config.device)
        weights_loss = torch.mean(weights * loss)
        return loss, weights_loss

    def get_DistributionalDQN_loss(self, batch, config, weights, gamma):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)
        done_batch = torch.cat([1 if s is not None else 0 for s in batch.next_state]).unsqueeze(1)

        delta_z = float(config.v_max - config.v_min) / (config.atom_size - 1)
        with torch.no_grad():
            next_action = self.policy_net(next_state_batch).argmax(1)
            next_dist = self.target_net.dist(next_state_batch)
            next_dist = next_dist[range(config.batch_size), next_action]

            t_z = reward_batch + (1 - done_batch) * gamma * self.support
            t_z = t_z.clamp(min=config.v_min, max=config.v_max)
            b = (t_z - config.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (config.batch_size - 1) * config.atom_size, config.batch_size
                ).long()
                .unsqueeze(1)
                .expand(config.batch_size, config.atom_size)
                .to(config.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=config.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.policy_net.dist(state_batch)
        log_p = torch.log(dist[range(config.batch_size), action_batch])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        # PrioritizedReplayMemory
        weights = torch.FloatTensor(np.array(weights).reshape(-1, 1)).to(config.device)
        weights_loss = torch.mean(weights * elementwise_loss)

        return elementwise_loss, weights_loss