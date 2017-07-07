import random as rd
from collections import deque

__author__ = 'Zhiwei'


class Memory(object):
    """
    Experience replay memory class

    Member function:
        constructor(memory_limit)

        add(element)

        sample(size)

    Instance:
        _memory_limit, int
        _memory, deque
        _size, memory size
    """

    def __init__(
            self,
            memory_limit
    ):
        """
        args:
            memory_limit, int, the maximal number of elements in memory """
        super(Memory, self).__init__()
        self._memory_limit = memory_limit

        self._memory = deque()
        self._size = 0

    def add(self, element):
        """
        args:
            element, tuple, usually should be (state, action, state', reward, done)

        usage:
            automatical pop out the oldest data when limit exceeds """
        self._memory.append(element)
        self._size += 1

        if self._size > self._memory_limit:
            self._memory.popleft()
            self._size -= 1

    def sample(self, batch_size):
        """
        args:
            batch_size, int

        usage:
            sample a batch from memory, whose number is batch_size """
        if batch_size <= 0 or batch_size > self._size:
            raise ValueError("sample return empty list")
            return []

        return rd.sample(self._memory, batch_size)


if __name__ == '__main__':
    import numpy as np
    a = np.empty((2, 3, 5))
    D = Memory(3)
    D.add(a)
    D.add(a)
    D.add(a)
    D.add(a)
    D.add(a)
    D.add(a)
    D.add(a)
    D.add(a)
    D.add(5)
    D.add(a)
    D.add(a)
    print D.sample(1)
    print len(D.memory)
