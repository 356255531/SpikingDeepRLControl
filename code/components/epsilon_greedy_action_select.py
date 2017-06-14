__author__ = "zhiwei"

import numpy as np
from keras.utils import np_utils


def epsilon_greedy_action_select(
    DQN_Q_approximator,
    state,
    action_num,
    epsilon
):
    """
        Get the greedy action of a given state
        action: [0, 1] or [1, 0] (np array)
    """
    if np.random.random() < epsilon:
        action_idx = np.random.randint(3)
    else:
        dqn_output = DQN_Q_approximator.predict(state)
        action_idx = np.argmax(dqn_output)

    action = np.zeros(action_num)
    action[action_idx] = 1

    return action


def main():
    from dqn import DQN
    dqn = DQN(1, 3, nb_hidden=1000, decoder="decoder.npy")
    # for x in range(1000):
    #     print epsilon_greedy_action_select(dqn, np.array([-0.1]), 3, 0.1)
    print dqn.predict(np.array([-1]))


if __name__ == '__main__':
    main()
