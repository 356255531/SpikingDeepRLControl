import numpy as np
import pdb


def batch_parser(batch):
    states = np.array([batch_instance[0] for batch_instance in batch])
    actions = np.array([batch_instance[1] for batch_instance in batch])
    rewards = np.array([batch_instance[2] for batch_instance in batch])
    states_bar = np.array([batch_instance[3] for batch_instance in batch])
    dones = np.array([batch_instance[4] for batch_instance in batch])

    return states, actions, rewards, states_bar, dones


def train_network(
        DQN,
        batch,
        discount_factor=0.9):
    states, actions, rewards, states_bar, dones = batch_parser(batch)
    states_bar_predict_val = DQN.predict(states_bar)
    target_q_func = []
    for idx, done in enumerate(dones):
        if done:
            target_q_func.append(rewards[idx])
        else:
            target_q_func.append(rewards[idx] + discount_factor * np.max(states_bar_predict_val[idx]))
    cost = DQN.train_network(states, actions, target_q_func)
    return cost


def main():
    a = []
    for x in xrange(1, 10):
        a.append(np.array([0, 1]))
    a = np.array(a)
    print a.shape


if __name__ == '__main__':
    main()
