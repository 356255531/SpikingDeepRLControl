import numpy as np
from copy import deepcopy


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


def train_network(
        DQN,
        batch,
        bellman_factor=0.9,
        learning_rate=0.01,
        batch_size=1
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
    states = states[:, None, :]
    states_bar_predict_val = DQN.predict(states, minibatch_size=batch_size)

    for idx, done in enumerate(dones):
        if done:
            for action_idx, action in enumerate(actions[idx]):
                states_bar_predict_val[idx][action_idx * 3 + action] = rewards[idx]
        else:
            for action_idx, action in enumerate(actions[idx]):
                td_error = rewards[idx] + bellman_factor * np.max(states_bar_predict_val[idx]) - \
                    states_bar_predict_val[idx][action_idx * 3 + action]
                states_bar_predict_val[idx][action_idx * 3 + action] = states_bar_predict_val[idx][action_idx * 3 + action] + learning_rate * td_error

    DQN.training(minibatch_size=1,
                 train_whole_dataset=states,
                 train_whole_labels=states_bar_predict_val[:, None, :],
                 num_epochs=1,
                 pre_train_weights='saved_weights/'
                 )


def main():
    a = []
    for x in xrange(1, 10):
        a.append(np.array([0, 1]))
    a = np.array(a)
    print a.shape


if __name__ == '__main__':
    main()
