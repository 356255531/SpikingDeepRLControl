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


def action_to_idx(actions, dim, batch_size):
    actions_idx = np.zeros([batch_size, 3 ** dim])
    for idx, action in enumerate(actions):
        sum = 3 * actions[idx][0]
        sum += actions[idx][1]
        actions_idx[idx][int(sum)] = 1

    return actions_idx


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
    states_bar_predict_val = DQN.predict(states_bar)
    target_q_func = []  # ann
    for idx, done in enumerate(dones):
        if done:
            target_q_func.append(rewards[idx])
        else:
            target_q_func.append(
                rewards[idx] + bellman_factor * np.max(states_bar_predict_val[idx]))

    batch_size = states.shape[0]

    actions_idx = action_to_idx(actions, dim, batch_size)

    cost = DQN.train_network(states, actions_idx, target_q_func)  # ann

    return cost


def main():
    a = []
    for x in xrange(1, 10):
        a.append(np.array([0, 1]))
    a = np.array(a)
    print a.shape


if __name__ == '__main__':
    main()
