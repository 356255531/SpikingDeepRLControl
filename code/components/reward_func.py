import numpy as np


class Reward(object):
    """docstring for Reward"""

    def __init__(self):
        super(Reward, self).__init__()

    def evlt(self, previous_state, state, goal_state):
        if np.array_equal(state, goal_state):
            return 10

        return -0.1


def main():
    reward = Reward()
    print reward.evlt((-1, 1), (1, 1), (0, 1))


if __name__ == '__main__':
    main()
