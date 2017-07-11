# Import Classes
from components import StateActionSpace_RobotArm, Reward, Goal
from components import RobotArmEnv

# Import libs
import numpy as np
import argparse
import nengo
import pytry
import h5py

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
    def __init__(self, actions, env,
                 name="Robot_Arm", mean_solved=0,
                 mean_cancel=0, max_eps=20000, max_trials_per_ep=100,
                 epsilon=0.9, b_render=True):
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
        self.b_render = b_render

        self.epsilon = epsilon
        self.decay = 0.9999

        # self.all_rewards = []

        super(Nengo_Arm_Sim, self).__init__(label="fuck", output=self.tick,
                                            size_in=9, size_out=3)

        # initialize openai gym environment
        self.env = env
        self.env.reset()

    def tick(self, t, x):
        if self.done or self.reached_max_trials:
            if self.reached_max_trials:
                print 'reset env because current episode took to much trials to complete (avoid unfinite loop)'
            self.reset()

        self.num_trials += 1
        if self.num_trials > self.max_trials_per_ep:
            self.reached_max_trials = True

        action_idx = np.argmax(x)
        action = np.array([])
        action = np.append(action, action_idx / 3)
        action_idx /= 3
        action = np.append(action, action_idx)

        ob, reward, self.done = self.env.step(action)

        rval = [item for item in ob]
        rval.append(reward)

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
                # print 'mean reward over last 100 episodes: ', self.mean_reward
                if self.mean_reward > self.mean_solved:
                    self.solved = True
                    print 'solved problem after ', self.num_eps, ' episodes with mean'
                elif self.mean_reward <= self.mean_cancel:
                    self.cancel = True
            self.num_eps += 1

            # if self.num_eps % 5 == 0:
            self.epsilon *= self.decay
            print '---------------current epsilon:---------------', self.epsilon

            self.num_trials = 0
            self.reached_max_trials = False
            self.env.reset()
            if self.num_eps > 15000:
                self.env._arm._if_visual = True
        else:
            if self.reached_max_eps:
                print 'reached maximal number of episodes, so close env unsolved'
            if self.cancel:
                print 'cancel due to poor learning performance'
            self.env.reset()


class QLearn(nengo.Network):
    def __init__(self, aigym, t_past=0.1, t_now=0.005,
                 gamma=0.9, init_state=np.zeros(9),
                 learning_rate=1e-3,
                 ):
        super(QLearn, self).__init__()

        with self:

            self.save_weight_path = '/home/huangbo/Desktop/weights/model.h5'

            self.desired_action = nengo.Ensemble(n_neurons=300, dimensions=9, radius=2.0)
            nengo.Connection(self.desired_action, self.desired_action, synapse=0.1)
            self.state = nengo.Ensemble(n_neurons=300, dimensions=2, radius=1.5)

            def selection(t, x):
                result = np.zeros(9)
                if np.random.uniform() < aigym.epsilon:

                    if np.random.uniform() < 0.5:
                        result[np.random.randint(0, 9)] = 1
                    else:
                        action = aigym.env.get_jacobi_action()
                        sum = action[0] * 3 + action[1]
                        result[int(sum)] = 1
                else:
                    choice = np.argmax(x)
                    result[choice] = 1

                # if aigym.num_eps > 100 and aigym.num_eps % 100 == 0:
                #     print '-------------------start to save model----------------------------------'
                #     print 'current step', aigym.num_eps

                #     self.sim = nengo.Simulator(ArmTrial.model)
                #     self.sim.run(time_in_seconds=20)
                #     with h5py.File(self.save_weight_path, 'w') as hf:
                #         hf.create_dataset('weights',
                #                           data=self.sim.data[self.conn_p][len(self.sim.trange()) - 1, :, :],
                #                           compression="gzip",
                #                           compression_opts=9)

                #     print '-------------------model is saved----------------------------------'

                return result

            self.select = nengo.Node(selection, size_in=9)
            nengo.Connection(self.select, self.desired_action)
            self.q = nengo.Node(None, size_in=9)

            def initial_q(state):
                return init_state

            self.conn = nengo.Connection(self.state,
                                         self.q,
                                         function=initial_q,
                                         learning_rule_type=nengo.PES(learning_rate=learning_rate,
                                                                      pre_tau=t_past)
                                         )

            self.conn_p = nengo.Probe(self.conn, 'weights')

            nengo.Connection(self.q, self.select)
            self.reward = nengo.Node(None, size_in=1)

            self.reward_array = nengo.Node(lambda t, x: x[:-1] * x[-1], size_in=10)
            nengo.Connection(self.reward, self.reward_array[-1])
            nengo.Connection(self.select, self.reward_array[:-1])

            self.error = nengo.Node(None, size_in=9)

            nengo.Connection(self.reward_array, self.error, synapse=t_past)
            nengo.Connection(self.q, self.error, synapse=t_now, transform=gamma)
            nengo.Connection(self.q, self.error, synapse=t_past, transform=-1)
            nengo.Connection(self.error, self.conn.learning_rule, transform=-1)

        nengo.Connection(aigym[:-1], self.state)
        nengo.Connection(aigym[-1], self.reward)
        nengo.Connection(self.desired_action, aigym)


class ArmTrial(pytry.NengoTrial):
    def params(self):
        # self.param('number of neurons in state', N_state=500)
        self.param('maximal number of epochs', max_eps=20000)
        self.param('mean when the problem is considered solved', mean_solved=-110)
        self.param('past time interval for q-update', t_past=0.1)
        self.param('now time interval for q-update', t_now=0.005)
        self.param('gamma parameter for q-update', gamma=0.9)
        self.param('initialization of q-values', init_state=[0, 0, 0])
        self.param('learning rate of the PES learning connection', learning_rate=1e-4)

    def model(self, p):
        self.model = nengo.Network(seed=2)
        with self.model:
            resolution_in_degree = 10 * np.ones(2)  # Discretization Resolution in Degree
            state_action_space = StateActionSpace_RobotArm(resolution_in_degree)  # Encode the joint to state

            reward_func = Reward()  # The rule of reward function
            goal_func = Goal((-3, 0))
            env = RobotArmEnv(
                state_action_space,
                reward_func,
                goal_func,
                if_simulator=True,
                if_visual=False,
                dim=2
            )

            self.mc = Nengo_Arm_Sim([0, 1, 2], env, mean_solved=0, mean_cancel=-500, max_eps=p.max_eps)
            self.ql = QLearn(aigym=self.mc,
                             t_past=p.t_past,
                             t_now=p.t_now,
                             gamma=p.gamma,
                             init_state=np.zeros(9),
                             learning_rate=p.learning_rate
                             )

        return self.model

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
