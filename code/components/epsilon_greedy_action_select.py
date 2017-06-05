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
        state: 4 frames
        action: [0, 1] or [1, 0] (np array)
    """
    dqn_output = DQN_Q_approximator.predict(np_utils.to_categorical(state, 36))
    if np.random.random() < epsilon:
        action_idx = np.random.randint(3)
        # action = np.zeros(action_num)
        # action[action_idx] = 1
    else:
        action_idx = np.argmax(dqn_output)

    return action_idx, dqn_output[0]
