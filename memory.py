from collections import namedtuple, deque
import random

from sumtree import SumTree, MinTree


Transition = namedtuple('Transion', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, capacity, alpha):
        super(PrioritizedReplayMemory, self).__init__(capacity)
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.alpha = alpha

        tree_capacity = 1
        while tree_capacity <= self.capacity:
            tree_capacity *= 2

        self.sum_tree = SumTree(tree_capacity)
        self.min_tree = MinTree(tree_capacity)

    def push(self, *args):
        super().push(*args)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.capacity

    def _get_indices(self, batch_size):
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _get_weight(self, idx, beta):
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight

    def sample(self, batch_size, beta):
        indices = self._get_indices(batch_size)

        memorys = [self.memory[ind] for ind in indices]
        weights = [self._get_weight(i, beta) for i in indices]

        return memorys, weights, indices

    def sample_from_indices(self, batch_size, beta, indices):
        indices = self._get_indices(batch_size)

        memorys = [self.memory[ind] for ind in indices]
        weights = [self._get_weight(i, beta) for i in indices]

        return memorys, weights, indices

    def update_priorities(self, indices, priorities):
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)


class NStepMemory(PrioritizedReplayMemory):
    def __init__(self, capacity, alpha, gamma, n_step):
        super(NStepMemory, self).__init__(capacity, alpha)
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)

    def push(self, *args):
        self.n_step_buffer.append(Transition(*args))

        if self.n_step_buffer[-1][-2] is not None and len(self.n_step_buffer) < self.n_step:
            return None

        G = 0
        for transion in reversed(list(self.n_step_buffer)):
            G = transion[-1] + self.gamma * G

        super().push(self.n_step_buffer[0][0], self.n_step_buffer[0][1], self.n_step_buffer[-1][2], G)
        tmp = self.n_step_buffer[0]
        self.n_step_buffer.clear()
        return tmp