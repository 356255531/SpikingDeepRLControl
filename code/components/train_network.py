import numpy as np


def batch_parser(batch):
    """
    args:
        mini batch, tuple

    usage:
        parse the mini-batch to different type batches """
    states = np.array([batch_instance[0] for batch_instance in batch])
    actions = np.array([batch_instance[1] for batch_instance in batch])
    rewards = np.array([batch_instance[2] for batch_instance in batch])
    states_bar = np.array([batch_instance[3] for batch_instance in batch])
    dones = np.array([batch_instance[4] for batch_instance in batch])

    return states, actions, rewards, states_bar, dones


def action_to_idx(actions, dim):
    action_in_idx = np.zeros([actions.shape[0], 3 ** dim])
    for idx, action in enumerate(actions):
        sum = 0
        for single_action_idx, single_action in enumerate(action):
            sum += 3 ** (dim - 1 - single_action_idx) * single_action
        action_in_idx[idx][sum] = 1

    return action_in_idx


def train_network(
        dim,
        DQN,
        batch,
        bellman_factor=0.9,
        learning_rate=10e-6
):
    """
    Note, dependency here needed
        1) DQN: Q-function approximator

    args:
        batch, tuple, mini-batch
        bellman_factor, float
        learning_rate, float

    usage:
        perform supervise learning using TD-Error """
    states, actions, rewards, states_bar, dones = batch_parser(batch)
    # states_bar_predict_val = DQN.predict(states)  # ann
    states_bar_predict_val = DQN.predict(states[:, None, :])

    # target_q_func = []  #ann
    # for idx, done in enumerate(dones):
    #     if done:
    #         target_q_func.append(rewards[idx])
    #     else:
    #         target_q_func.append(
    #             rewards[idx] + bellman_factor * np.max(states_bar_predict_val[idx]))

    for idx, done in enumerate(dones):
        if done:
            states_bar_predict_val[idx][actions[idx]] = rewards[idx]
        else:
            states_bar_predict_val[idx][actions[idx]] = rewards[idx] + \
                bellman_factor * np.max(states_bar_predict_val[idx])

    # actions = action_to_idx(actions, dim)  # ann

    # cost = DQN.train_network(states, actions, target_q_func)  # ann

    cost = DQN.train_network(states[:, None, :], states_bar_predict_val[:, None, :], 1)
    return cost


def main():
    a = []
    for x in xrange(1, 10):
        a.append(np.array([0, 1]))
    a = np.array(a)
    print a.shape


if __name__ == '__main__':
    main()
