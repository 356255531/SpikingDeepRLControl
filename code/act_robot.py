# Import Classes
from components import Memory, DQN, RobotArmEnv, StateActionSpace_RobotArm, Reward, Goal
# Import functions
from components import epsilon_greedy_action_select, train_network, epsilon_decay

import numpy as np

# Game set up
ACTION_NUM = 3

# Training set up
BATCH_SIZE = 1000
MEMERY_LIMIT = 1000

# RL parameters
BELLMAN_FACTOR = 0.9

# exploration setting
OBSERVE_PHASE = 1000

# epsilon setting
EPSILON_DISCOUNT = 0.9999
EPSILON_START = 0.1
EPSILON_FINAL = 0.0001


def train_dqn():
    epsilon = EPSILON_START

    state_action_space = StateActionSpace_RobotArm()
    reward_func = Reward()
    goal_func = Goal((180,), state_action_space)
    env = RobotArmEnv(state_action_space, reward_func, goal_func)
    env.init_game()
    done = False

    # Create memory_pool poolenv
    display_memory = Memory(MEMERY_LIMIT)

    # Create network object
    dqn = DQN(1, 3, nb_hidden=1000, decoder="decoder.npy")
    # dqn = Q_learning_network()
    # dqn.load_weights()

    # Q-Learning framework

    while 1:
        cost = 0
        total_step = 0
        num_episode = 0

        while total_step < OBSERVE_PHASE:
            state = env.init_game()
            done = False
            while not done:
                if total_step > OBSERVE_PHASE:
                    break

                total_step += 1

                action = epsilon_greedy_action_select(
                    dqn,
                    state,
                    ACTION_NUM,
                    epsilon
                )

                state_bar, reward, done = env.step(action)
                print "state", state, "reward", reward, "total step", total_step

                display_memory.add((state, action, reward, state_bar, done))

                state = state_bar

        batch = display_memory.sample(BATCH_SIZE)

        cost = train_network(
            dqn,
            batch,
            BELLMAN_FACTOR
        )

        print "reward: ", reward, " cost: ", cost, " action: ", np.argmax(action), " if game continue: ", not done, " epsilon: ", epsilon

        if 0 == ((num_episode + 1) % 1000):
            print "Cost is: ", cost
            dqn.save_weights(num_episode)  # save weights

        epsilon = epsilon_decay(epsilon, EPSILON_DISCOUNT, EPSILON_FINAL)


def main():
    train_dqn()


if __name__ == '__main__':
    train_dqn()
