import numpy as np


class Reward(object):
    """
    Reward function class

    Member function:
        evtl(previous_state, action, state, goal_state) """

    def __init__(self):
        super(Reward, self).__init__()

    def evlt(self, previous_state, action, state, goal_state):
        """
        args:
            previous_state, numpy array
            action, numpy array
            state, numpy array
            goal state, numpy array

        return:
            reward accordingto the setting above """
        if np.array_equal(state, goal_state):
            return 10

        return -0.1


def main():
    reward = Reward()
    print reward.evlt((-1, 1), (1, 1), (0, 1))


if __name__ == '__main__':
    main()
