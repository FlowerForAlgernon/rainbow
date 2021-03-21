import operator
from typing import Callable


class SegmentTree:
    def __init__(self, capacity: int, operation: Callable, init_value: float):
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."

        self.capacity = capacity
        # 这里存储树的数组长度为 2 * capacity，是因为根节点是在 1 位置
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        # 这里 start end 是指样本数组目标区间，node_start node_end 是指样本数组区间，node 是指存储树的数组的当前节点位置
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumTree(SegmentTree):
    def __init__(self, capacity: int):
        super(SumTree, self).__init__(capacity=capacity, operation=operator.add, init_value=0.0)

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1
        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinTree(SegmentTree):
    def __init__(self, capacity: int):
        super(MinTree, self).__init__(capacity=capacity, operation=min, init_value=float("inf"))

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinTree, self).operate(start, end)
