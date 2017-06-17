# Import Classes
from components import Memory, DQN, RobotArmEnv, StateActionSpace_RobotArm, Reward, Goal
# Import functions
from components import epsilon_greedy_action_select, train_network, epsilon_decay

# Import libs
import numpy as np
import argparse

# Argument parser
parser = argparse.ArgumentParser()

parser.add_argument("-sim", "--emulator", nargs="?", const=1, type=bool, help="Use a emulator [Y/n]", default=True)
parser.add_argument("-d", "--dimension", nargs="?", const=1, type=int, help="Dimension of a robot [default: 1]", default=1)
parser.add_argument("-t", "--train", nargs="?", const=1, type=bool, help="Train the robot arm [or acting: a]", default=True)
parser.add_argument("-p", "--path", nargs="?", const=1, type=str, help="path to decoder", default="saved_weights/")
parser.add_argument("-v", "--visualization", nargs="?", const=False, type=bool, help="If visulize robot arm action [y/N]", default=False)
parser.add_argument("-r", "--learning_rate", nargs="?", const=1, type=float, help="The learning rate", default=0.01)
parser.add_argument("-bf", "--bellman_factor", nargs="?", const=1, type=float, help="Bellman factor", default=0.9)
parser.add_argument("-l", "--memory_limit", nargs="?", const=1, type=int, help="The limit of display memory", default=1000)
parser.add_argument("-e", "--epsilon", nargs="?", const=1, type=float, help="Epsilon value of policy selection", default=0.1)
parser.add_argument("-ef", "--bellman_factor_final", nargs="?", const=1, type=float, help="Final epsilon value", default=0.0001)
parser.add_argument("-ed", "--epsilon_decay", nargs="?", const=1, type=float, help="Decay factor of epsilon", default=0.9999)

args = parser.parse_args()

# Action number
ACTION_NUM = 3 ** args.dimension  # Each joint has three actions


def train_dqn():
    epsilon = args.epsilon

    resolution_in_degree = 10 * np.ones(args.dimension)
    state_action_space = StateActionSpace_RobotArm(resolution_in_degree)

    reward_func = Reward()
    goal_func = Goal((180,), state_action_space)
    env = RobotArmEnv(
        state_action_space,
        reward_func,
        goal_func,
        if_emulator=True,
        if_visual=args.visualization,
        dim=args.dimension
    )

    state = env.init_game()
    done = False

    # Create memory_pool poolenv
    display_memory = Memory(args.memory_limit)

    # Create network object
    dqn = DQN(1, 3, nb_hidden=1000, decoder=args.path + "decoder.npy")
    # dqn = Q_learning_network()
    # dqn.load_weights()

    # Q-Learning framework

    while 1:
        cost = 0
        total_step = 0
        num_episode = 0

        while total_step < args.memory_limit:
            state = env.init_game()
            done = False
            count = 0
            while not done:
                if count > 200:
                    break
                count += 1

                if total_step > args.memory_limit:
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

                print "reward: ", reward, " action: ", np.argmax(action), " if game continue: ", not done, " epsilon: ", epsilon

        if args.train:
            batch = display_memory.sample(args.memory_limit)

            cost = train_network(
                dqn,
                batch,
                args.bellman_factor
            )

            print " cost: ", cost

        if 0 == ((num_episode + 1) % 1000):
            print "Cost is: ", cost
            dqn.save_weights(num_episode)  # save weights

        epsilon = epsilon_decay(epsilon, args.epsilon_decay, args.bellman_factor_final)
        num_episode += 1


def main():
    train_dqn()


if __name__ == '__main__':
    train_dqn()
