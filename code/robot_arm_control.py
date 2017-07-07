# Import Classes
from components import Memory
from components import ANN
from components import StateActionSpace_RobotArm, Reward, Goal
from components import RobotArmEnv
# Import functions
from components import epsilon_greedy_action_select, train_network, epsilon_decay

# Import libs
import numpy as np
import argparse

import pdb
import operator

# Argument parser
parser = argparse.ArgumentParser()

parser.add_argument("-sim", "--simulator", nargs="?", const=1,
                    type=bool, help="Use a simulator [Y/n]", default=True)
parser.add_argument("-d", "--dimension", nargs="?", const=1,
                    type=int, help="Dimension of a robot [default: 1]", default=1)
parser.add_argument("-t", "--train", nargs="?", const=1,
                    type=bool, help="Train the robot arm [or acting: a]", default=True)
parser.add_argument("-p", "--path", nargs="?", const=1, type=str, help="path to decoder",
                    default="saved_weights_ann/")
parser.add_argument("-v", "--visualization", nargs="?", const=False,
                    type=bool, help="If visulize robot arm action [y/N]", default=False)
parser.add_argument("-r", "--learning_rate", nargs="?", const=1,
                    type=float, help="The learning rate", default=10e-6)
parser.add_argument("-bs", "--batch_size", nargs="?", const=1,
                    type=int, help="The training batch size", default=128)
parser.add_argument("-bf", "--bellman_factor", nargs="?", const=1,
                    type=float, help="Bellman factor", default=0.9)
parser.add_argument("-l", "--memory_limit", nargs="?", const=1,
                    type=int, help="The limit of display memory", default=50000)
parser.add_argument("-eml", "--episode_max_len", nargs="?", const=1,
                    type=int, help="The maximal length of a episode", default=36)
parser.add_argument("-e", "--epsilon", nargs="?", const=1,
                    type=float, help="Epsilon value of policy selection", default=0.5)
parser.add_argument("-ef", "--epsilon_final", nargs="?", const=1,
                    type=float, help="Final epsilon value", default=0.05)
parser.add_argument("-ed", "--epsilon_decay", nargs="?", const=1,
                    type=float, help="Decay factor of epsilon", default=0.999)
parser.add_argument("-op", "--observation_phase", nargs="?", const=1,
                    type=float, help="In observation phase, no training behavior", default=10000)
parser.add_argument("-ep", "--exploration_phase", nargs="?", const=1, type=float,
                    help="In exploration phase, algorithm continue explore all possible actions",
                    default=100000)

args = parser.parse_args()


def train_dqn(
    if_simulator,
    joint_dim,
    if_train,
    weight_path,
    if_visual,
    learning_rate,
    batch_size,
    bellman_factor,
    memory_limit,
    episode_max_len,
    epsilon,
    epsilon_decay_factor,
    epsilon_final,
    observation_phase,
    exploration_phase
):
    resolution_in_degree = 10 * np.ones(
        args.dimension)  # Discretization Resolution in Degree
    state_action_space = StateActionSpace_RobotArm(
        resolution_in_degree)  # Encode the joint to state

    reward_func = Reward()  # The rule of reward function
    goal_func = Goal((-3, 0))
    env = RobotArmEnv(
        state_action_space,
        reward_func,
        goal_func,
        if_simulator=if_simulator,
        if_visual=if_visual,
        dim=joint_dim
    )

    # Create memory_pool
    display_memory = Memory(memory_limit)

    dqn = ANN(joint_dim, learning_rate)  # ann
    dqn.load_weights(weight_path, "dqn_weights")  # ann

    # Q-Learning framework

    total_step = 0
    num_episode = 0
    cost = float("inf")

    while 1:
        num_episode += 1

        state = env.init_game()
        done = False

        episode_step = 0
        total_reward = 0
        while episode_step < episode_max_len and not done:
            action = epsilon_greedy_action_select(
                env,
                dqn,
                state,
                joint_dim,
                epsilon,
                batch_size
            )

            state_bar, reward, done = env.step(action)
            total_reward += reward
            episode_step += 1
            total_step += 1
            print "state: ", state, " action: ", action, \
                "reward: ", reward, \
                " if game continue: ", not done, \
                " epsilon: ", epsilon, \
                " cost: ", cost, " total_step: ", total_step, \
                " num_episode: ", num_episode

            display_memory.add((state, action,
                                reward, state_bar, done))

            state = state_bar

            if if_train and total_step > observation_phase:
                batch = display_memory.sample(batch_size)

                cost = train_network(  # Training Step
                    joint_dim,
                    dqn,
                    batch,
                    bellman_factor,
                    learning_rate
                )

        if total_step > observation_phase and (num_episode - 1) % 100 == 1:  # ann
            dqn.save_weights(num_episode, weight_path, "dqn_weights")  # ann

        if total_step <= exploration_phase:
            if total_step > observation_phase:
                epsilon = epsilon_decay(
                    epsilon, epsilon_decay_factor, epsilon_final)
        else:
            epsilon = epsilon_final
    # epsilon = 1
    # done = True
    # q_table = {}
    # for i in xrange(100000):
    #     state = env.init_game()
    #     q_table[tuple(state)] = {}
    #     for j in xrange(9):
    #         q_table[tuple(state)][j] = np.random.rand()

    # while 1:
    #     if done:
    #         state = env.init_game()
    #         done = False
    #     if np.random.rand() < epsilon:
    #         action = np.random.randint(3, size=[2])
    #     else:
    #         action = max(q_table[tuple(state)].iteritems(), key=operator.itemgetter(1))[0]
    #         action = np.array([action / 3, action % 3])

    #     state_bar, reward, done = env.step(action)
    #     print state, state_bar, action, done, epsilon, (env._arm.read_end_coor()[0] + 3) ** 2 + (env._arm.read_end_coor()[1]) ** 2
    #     q_table[tuple(state)][max(q_table[tuple(state)].iteritems(), key=operator.itemgetter(1))[0]] += \
    #         0.1 * (reward + 0.5 * max(q_table[tuple(state)], key=q_table.get) - q_table[tuple(state)][max(q_table[tuple(state)].iteritems(), key=operator.itemgetter(1))[0]])
    #     state = state_bar
    #     epsilon *= 0.99999


def main():
    if_simulator = args.simulator
    joint_dim = args.dimension
    if_train = args.train
    weight_path = args.path
    if_visual = args.visualization
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    bellman_factor = args.bellman_factor
    memory_limit = args.memory_limit
    episode_max_len = args.episode_max_len
    epsilon = args.epsilon
    epsilon_decay_factor = args.epsilon_decay
    epsilon_final = args.epsilon_final
    observation_phase = args.observation_phase
    exploration_phase = args.exploration_phase
    train_dqn(
        if_simulator,
        joint_dim,
        if_train,
        weight_path,
        if_visual,
        learning_rate,
        batch_size,
        bellman_factor,
        memory_limit,
        episode_max_len,
        epsilon,
        epsilon_decay_factor,
        epsilon_final,
        observation_phase,
        exploration_phase
    )


if __name__ == '__main__':
    main()
