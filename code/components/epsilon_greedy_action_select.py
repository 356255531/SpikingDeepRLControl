__author__ = "zhiwei"

import numpy as np


def epsilon_greedy_action_select(
    DQN_Q_approximator,
    state,
    dim,
    epsilon,
    batch_size
):
    """
    note, dependency needed here
        1) DQN_Q_approximator:  The Q-function approximator, should be
                                able to do prediction with given state

    args:
            state, numpy array
            dim, int, joint dimension
            epsilon, int

    usage:
        return an action using epsilon-greedy search """
    if np.random.random() < epsilon:
        return np.random.randint(3, size=(dim,))
    else:
        # states = np.repeat(np.array([state]), batch_size, axis=0) #snn
        # dqn_output = DQN_Q_approximator.predict(states[:, None, :])[0, :]

        dqn_output = DQN_Q_approximator.predict(np.array([state]))  # ann
        action = np.array([], dtype=int)
        for i in xrange(dim):
            action = np.append(
                action,
                np.argmax(dqn_output[i * 3:(i + 1) * 3])
            )
        return action


def main():
    from dqn import DQN
    dqn = DQN(1, 3, nb_hidden=1000, decoder="decoder.npy")
    # for x in range(1000):
    #     print epsilon_greedy_action_select(dqn, np.array([-0.1]), 3, 0.1)
    print dqn.predict(np.array([0]))


if __name__ == '__main__':
    main()
