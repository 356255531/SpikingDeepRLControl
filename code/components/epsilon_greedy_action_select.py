import numpy as np
__author__ = "zhiwei"


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
    if np.random.random() < epsilon:
        action_idx = np.random.randint(action_num)
        action = np.zeros(action_num)
        action[action_idx] = 1
        return action

    Q_func = DQN_Q_approximator.predict(trans_state2input_1d(state))

    action = np.zeros(action_num)
    action[np.argmax(Q_func)] = 1

    return action


def trans_state2input_1d(state):
    state_binary = bin(np.array(state))
    return state_binary
