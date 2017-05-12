#!/usr/bin/python
import collections
import numpy as np
import sys


BATCH_SIZE = 64
BUFFER_SIZE = 1000000


class ReplayMemory:
    def __init__(self, max_size=BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=max_size)
        self.size = len(self.buffer)

    def get_buffer_size(self):
        # return current queue size
        return self.size

    def get_minibatch_samples(self, number_of_samples=BATCH_SIZE):
        if self.size<BATCH_SIZE:
            return None # wait for more samples in buffer
        else:
            ids = np.random.choice(np.arange(self.size), number_of_samples)
            return np.array([self.buffer[i] for i in ids])

    def add_sample(self, sample):
        self.buffer.append(sample)
        self.size = len(self.buffer)