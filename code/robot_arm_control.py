# Import Classes
from components import StateActionSpace_RobotArm, Reward, Goal
from components import RobotArmEnv

# Import libs
import numpy as np
import argparse
import nengo
import pytry

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


class Nengo_Arm_Sim(nengo.Node):
    def __init__(self, actions, env, mean_solved=-110, mean_cancel=-500, max_eps=10000, max_trials_per_ep=36):
        self.actions = actions
        self.done = False
        self.solved = False
        self.max_trials_per_ep = max_trials_per_ep
        self.reached_max_trials = False
        self.reached_max_eps = False
        self.num_trials = 0
        self.num_eps = 0
        self.last_hundred_rewards = [0] * 100
        self.mean_reward = -10000
        self.mean_solved = mean_solved
        self.mean_cancel = mean_cancel
        self.cancel = False
        self.max_eps = max_eps
        # self.all_rewards = []

        super(Nengo_Arm_Sim, self).__init__(label="fuck", output=self.tick,
                                            size_in=len(self.actions), size_out=2)

        # initialize openai gym environment
        self.env = env

    def tick(self, t, x):
        self.num_trials += 1
        if self.num_trials > self.max_trials_per_ep:
            self.reached_max_trials = True
        action = np.argmax(x)
        ob, reward, done, _ = self.env.step([action])
        rval = [item for item in ob]
        rval.append(reward)
        if self.b_render:
            self.env.render()

        self.position = ob[0]

        if self.position > 0.5:
            self.done = True

        if self.done or self.reached_max_trials:
            if self.reached_max_trials:
                print 'reset env because current episode took to much trials to complete (avoid unfinite loop)'
            self.reset()

        return rval

    def reset(self):
        if not self.reached_max_eps and not self.solved and not self.cancel:
            print "resetting after episode ", self.num_eps
            if self.num_eps > self.max_eps:
                self.reached_max_eps = True
            self.done = False
            ep_reward = -1 * self.num_trials
            print 'reward of this episode: ', ep_reward
            # self.all_rewards.append(ep_reward)
            if self.num_eps < 100:
                self.last_hundred_rewards[self.num_eps] = ep_reward
            else:
                self.last_hundred_rewards[:-1] = self.last_hundred_rewards[1:]
                self.last_hundred_rewards[-1] = ep_reward
                self.mean_reward = np.mean(self.last_hundred_rewards)
                print 'mean reward over last 100 episodes: ', self.mean_reward
                if self.mean_reward > self.mean_solved:
                    self.solved = True
                    print 'solved problem after ', self.num_eps, ' episodes with mean'
                elif self.mean_reward <= self.mean_cancel:
                    self.cancel = True
            self.num_eps += 1
            self.num_trials = 0
            self.reached_max_trials = False
            self.env.reset()
        else:
            if self.reached_max_eps:
                print 'reached maximal number of episodes, so close env unsolved'
            if self.cancel:
                print 'cancel due to poor learning performance'
            self.env.close()


class QLearn1(nengo.Network):
    def __init__(self, aigym, t_past=0.1, t_now=0.005, gamma=0.9, init_state=[0, 0, 0], learning_rate=1e-4):
        super(QLearn1, self).__init__()
        with self:

            self.desired_action = nengo.Ensemble(n_neurons=300, dimensions=3, radius=2.0)
            nengo.Connection(self.desired_action, self.desired_action, synapse=0.1)

            self.state = nengo.Ensemble(n_neurons=300, dimensions=1, radius=1.5)

            def selection(t, x):
                choice = np.argmax(x)

                result = np.zeros(3)
                result[choice] = 1
                return result

            self.select = nengo.Node(selection, size_in=1)

            nengo.Connection(self.select, self.desired_action)

            self.q = nengo.Node(None, size_in=3)

            def initial_q(state):
                return init_state

            self.conn = nengo.Connection(self.state, self.q, function=initial_q,
                                         learning_rule_type=nengo.PES(learning_rate=learning_rate, pre_tau=t_past))
            nengo.Connection(self.q, self.select)

            self.reward = nengo.Node(None, size_in=1)

            self.reward_array = nengo.Node(lambda t, x: x[:-1] * x[-1], size_in=3)
            nengo.Connection(self.reward, self.reward_array[-1])
            nengo.Connection(self.select, self.reward_array[:-1])

            self.error = nengo.Node(None, size_in=3)

            nengo.Connection(self.reward_array, self.error, synapse=t_past)
            nengo.Connection(self.q, self.error, synapse=t_now, transform=gamma)
            nengo.Connection(self.q, self.error, synapse=t_past, transform=-1)

            nengo.Connection(self.error, self.conn.learning_rule, transform=-1)

        nengo.Connection(aigym[:-1], self.state)
        nengo.Connection(aigym[-1], self.reward)
        nengo.Connection(self.desired_action[0], aigym[0])
        nengo.Connection(self.desired_action[1], aigym[2])


class ArmTrial(pytry.NengoTrial):
    def params(self):
        # self.param('number of neurons in state', N_state=500)
        self.param('maximal number of epochs', max_eps=5000)
        self.param('mean when the problem is considered solved', mean_solved=-110)
        self.param('past time interval for q-update', t_past=0.1)
        self.param('now time interval for q-update', t_now=0.005)
        self.param('gamma parameter for q-update', gamma=0.9)
        self.param('initialization of q-values', init_state=[0, 0, 0])
        self.param('learning rate of the PES learning connection', learning_rate=1e-4)

    def model(self, p):
        model = nengo.Network(seed=2)
        with model:
            resolution_in_degree = 10 * np.ones(1)  # Discretization Resolution in Degree
            state_action_space = StateActionSpace_RobotArm(resolution_in_degree)  # Encode the joint to state

            reward_func = Reward()  # The rule of reward function
            goal_func = Goal((-3, 0))
            env = RobotArmEnv(
                state_action_space,
                reward_func,
                goal_func,
                if_simulator=True,
                if_visual=True,
                dim=1
            )
            self.mc = Nengo_Arm_Sim([0, 1, 2], env, mean_solved=p.mean_solved, mean_cancel=-500, max_eps=p.max_eps)
            #  self.ql = QLearn(aigym=self.mc, t_past=p.t_past, t_now=p.t_now, gamma=p.gamma, init_state = p.init_state)
            self.ql = QLearn1(aigym=self.mc, t_past=p.t_past, t_now=p.t_now, gamma=p.gamma, init_state=p.init_state, learning_rate=p.learning_rate)

        return model

    def evaluate(self, p, sim, plt):
        while not self.mc.reached_max_eps and not self.mc.cancel:
            sim.run(2)
        return dict(solved=self.mc.solved, episodes=self.mc.num_eps, last_hundred_rewards=self.mc.last_hundred_rewards)


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
    for t_past in [0.1]:
        for t_now in [0.005]:
            for gamma in [0.9]:
                # for init_state in [[0,0,0], [0.5,0, 0.5]]:
                for init_state in [[0, 0, 0], [0.5, 0.5, 0.5]]:
                    for learning_rate in [1e-3, 1e-4, 1e-5]:
                        ArmTrial().run(t_past=t_past, t_now=t_now, gamma=gamma, init_state=init_state, verbose=False)


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
