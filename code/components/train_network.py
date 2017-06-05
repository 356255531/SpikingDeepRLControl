import numpy as np
from keras.utils import np_utils


def batch_parser(batch):
    states = np.array([batch_instance[0] for batch_instance in batch])
    actions = np.array([batch_instance[1] for batch_instance in batch])
    rewards = np.array([batch_instance[2] for batch_instance in batch])
    states_bar = np.array([batch_instance[3] for batch_instance in batch])
    dones = np.array([batch_instance[4] for batch_instance in batch])
    dqn_outputs = np.array([batch_instance[5] for dqn_output_instance in batch])

    return states, actions, rewards, states_bar, dones, dqn_outputs


def train_network(
        DQN,
        batch,
        discount_factor=0.9):
    states, actions, rewards, states_bar, dones, dqn_outputs_previous = batch_parser(batch)
    dqn_outputs = dqn_outputs_previous
    states_bar_predict_val = DQN.predict(np_utils.to_categorical(states_bar, 36))
    for idx, done in enumerate(dones):
        if done:
            dqn_outputs[idx][actions[idx]] = rewards[idx]
        else:
            dqn_outputs[idx][actions[idx]] = rewards[idx] + discount_factor * np.max(states_bar_predict_val[idx])

    DQN.train_network(np_utils.to_categorical(states, 36), dqn_outputs, 1)

    return np.linalg.norm(dqn_outputs_previous - dqn_outputs)


def main():
    a = []
    for x in xrange(1, 10):
        a.append(np.array([0, 1]))
    a = np.array(a)
    print a.shape


if __name__ == '__main__':
    main()
