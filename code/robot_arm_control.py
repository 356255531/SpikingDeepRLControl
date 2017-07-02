# Import Classes
from components import Memory
from components import ANN, SNN
from components import StateActionSpace_RobotArm, Reward, Goal
from components import RobotArmEnv
# Import functions
from components import epsilon_greedy_action_select, train_network, epsilon_decay

# Import libs
import numpy as np
import argparse

# Argument parser
parser = argparse.ArgumentParser()

parser.add_argument("-sim", "--emulator", nargs="?", const=1,
                    type=bool, help="Use a emulator [Y/n]", default=True)
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
                    type=int, help="The maximal length of a episode", default=100)
parser.add_argument("-e", "--epsilon", nargs="?", const=1,
                    type=float, help="Epsilon value of policy selection", default=0.5)
parser.add_argument("-ef", "--epsilon_final", nargs="?", const=1,
                    type=float, help="Final epsilon value", default=0.05)
parser.add_argument("-ed", "--epsilon_decay", nargs="?", const=1,
                    type=float, help="Decay factor of epsilon", default=0.9999)
parser.add_argument("-ob", "--observation_phase", nargs="?", const=1,
                    type=float, help="In observation phase, no training behavior", default=10000)
parser.add_argument("-ep", "--exploration_phase", nargs="?", const=1, type=float,
                    help="In exploration phase, algorithm continue explore all possible actions",
                    default=100000)

args = parser.parse_args()


def train_dqn(
    if_emulator,
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
    goal_func = Goal((180,), state_action_space)
    env = RobotArmEnv(
        state_action_space,
        reward_func,
        goal_func,
        if_emulator=if_emulator,
        if_visual=if_visual,
        dim=joint_dim
    )

    # Create memory_pool poolenv
    display_memory = Memory(memory_limit)

    # Create network object
    dqn = ANN(joint_dim, learning_rate)
    dqn.load_weights(weight_path)

    # Q-Learning framework

    total_step = 0
    num_episode = 0
    cost = float("inf")
    total_reward_previous = -100
    while 1:
        num_episode += 1
        state = env.init_game()
        done = False

        episode_step = 0
        total_reward = 0
        while episode_step < episode_max_len and not done:
            action = epsilon_greedy_action_select(
                dqn,
                state,
                joint_dim,
                epsilon
            )

            state_bar, reward, done = env.step(action)
            total_reward += reward
            episode_step += 1
            total_step += 1
            print "reward: ", reward, " action: ", action, \
                " if game continue: ", \
                not done, " epsilon: ", epsilon, \
                " cost: ", cost, " total_reward: ", total_reward_previous

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

        total_reward_previous = total_reward

        if (total_step - 1) % 1000 == 1:
            dqn.save_weights(num_episode, weight_path)

        if total_step <= exploration_phase:
            if total_step > observation_phase:
                epsilon = epsilon_decay(
                    epsilon, epsilon_decay_factor, epsilon_final)
        else:
            epsilon = epsilon_final


def main():
    if_emulator = args.emulator
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
        if_emulator,
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
