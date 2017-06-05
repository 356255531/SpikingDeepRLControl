# Import Classes
from components import Memory, DQN, RobotArmEnv, StateActionSpace_RobotArm, Reward, Goal
# Import functions
from components import epsilon_greedy_action_select, train_network, epsilon_decay

import numpy as np

# Game set up
ACTION_NUM = 3

# Training set up
# BATCH_SIZE = 50000
MEMERY_LIMIT = 100000

# RL parameters
BELLMAN_FACTOR = 0.9

# exploration setting
OBSERVE_PHASE = 50000
EXPLORE_PHASE = 200000

# epsilon setting
EPSILON_DISCOUNT = 0.9999
EPSILON_START = 0.1
EPSILON_FINAL = 0.0001


def train_dqn():
    epsilon = EPSILON_START

    state_action_space_obj = StateActionSpace_RobotArm()
    reward_func_obj = Reward()
    goal_func_obj = Goal((18,))
    env = RobotArmEnv(state_action_space_obj, reward_func_obj, goal_func_obj)
    env.init_game()
    done = False

    # Create memory_pool poolenv
    display_memory = Memory(MEMERY_LIMIT)

    # Create network object
    dqn = DQN(36, 3, nb_hidden=1000, decoder="decoder.npy")

    # Q-Learning framework
    cost = 0
    total_step = 0
    num_episode = 0
    while 1:
        num_episode += 1

        state = env.init_game()

        done = False

        while not done:
            total_step += 1
            print total_step

            action, q_func = epsilon_greedy_action_select(
                dqn,
                state,
                ACTION_NUM,
                epsilon
            )

            state_bar, reward, done = env.step(np.array([action]))

            display_memory.add((state, action, reward, state_bar, done, q_func))

            if total_step > OBSERVE_PHASE:
                import pdb
                pdb.set_trace()
                batch = display_memory.sample(BATCH_SIZE)

                cost = train_network(
                    dqn,
                    batch,
                    BELLMAN_FACTOR
                )
                print state
                print "reward: ", reward, " cost: ", cost, " action: ", np.argmax(action), " if continue: ", not done, " epsilon: ", epsilon

            state = state_bar

        if 0 == ((num_episode + 1) % 1000):
            print "Cost is: ", cost
            # dqn.save_weights(num_episode)  # save weights

        if total_step > EXPLORE_PHASE:
            epsilon = EPSILON_FINAL
        elif total_step > OBSERVE_PHASE:
            epsilon = epsilon_decay(epsilon, EPSILON_DISCOUNT, EPSILON_FINAL)


def main():
    train_dqn()


if __name__ == '__main__':
    train_dqn()
