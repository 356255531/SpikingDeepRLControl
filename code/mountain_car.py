import gym
import nengo
import numpy as np
import pytry


class OpenAIGym(nengo.Node):
    def __init__(self, actions, name='MountainCar-v0', mean_solved=-110, mean_cancel=-500, max_eps=10000, max_trials_per_ep=2000, b_render=True):
        self.actions = actions
        self.name = name
        self.b_render = b_render
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

        super(OpenAIGym, self).__init__(label=self.name, output=self.tick,
                                        size_in=len(self.actions), size_out=3)

        # initialize openai gym environment
        self.env = gym.make(self.name)
        self.env.reset()

    def tick(self, t, x):
        self.num_trials += 1
        if self.num_trials > self.max_trials_per_ep:
            self.reached_max_trials = True
        action = np.argmax(x)
        ob, reward, done, _ = self.env.step(action)
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


class QLearn(nengo.Network):
    def __init__(self, aigym, t_past=0.1, t_now=0.005, gamma=0.9, init_state=[0, 0, 0]):
        super(QLearn, self).__init__()
        with self:

            self.desired_action = nengo.Ensemble(n_neurons=300, dimensions=3, radius=2.0)
            nengo.Connection(self.desired_action, self.desired_action, synapse=0.1)

            self.state = nengo.Ensemble(n_neurons=300, dimensions=2, radius=1.5)

            def selection(t, x):
                choice = np.argmax(x)

                result = np.zeros(3)
                result[choice] = 1
                return result

            self.select = nengo.Node(selection, size_in=3)

            nengo.Connection(self.select, self.desired_action)

            self.q = nengo.Node(None, size_in=3)

            def initial_q(state):
                return init_state

            self.conn = nengo.Connection(self.state, self.q, function=initial_q,
                                         learning_rule_type=nengo.PES(pre_tau=t_past))
            nengo.Connection(self.q, self.select)

            self.reward = nengo.Node(None, size_in=1)

            self.reward_array = nengo.Node(lambda t, x: x[:-1] * x[-1], size_in=4)
            nengo.Connection(self.reward, self.reward_array[-1])
            nengo.Connection(self.select, self.reward_array[:-1])

            self.error = nengo.Node(None, size_in=3)

            nengo.Connection(self.reward_array, self.error, synapse=t_past)
            nengo.Connection(self.q, self.error, synapse=t_now, transform=gamma)
            nengo.Connection(self.q, self.error, synapse=t_past, transform=-1)

            nengo.Connection(self.error, self.conn.learning_rule, transform=-1)

        nengo.Connection(aigym[:-1], self.state)
        nengo.Connection(aigym[-1], self.reward)
        nengo.Connection(self.desired_action, aigym)


class QLearn1(nengo.Network):
    def __init__(self, aigym, t_past=0.1, t_now=0.005, gamma=0.9, init_state=[0, 0], learning_rate=1e-4):
        super(QLearn1, self).__init__()
        with self:

            self.desired_action = nengo.Ensemble(n_neurons=300, dimensions=2, radius=2.0)
            nengo.Connection(self.desired_action, self.desired_action, synapse=0.1)

            self.state = nengo.Ensemble(n_neurons=300, dimensions=2, radius=1.5)

            def selection(t, x):
                choice = np.argmax(x)

                result = np.zeros(2)
                result[choice] = 1
                return result

            self.select = nengo.Node(selection, size_in=2)

            nengo.Connection(self.select, self.desired_action)

            self.q = nengo.Node(None, size_in=2)

            def initial_q(state):
                return init_state

            self.conn = nengo.Connection(self.state, self.q, function=initial_q,
                                         learning_rule_type=nengo.PES(learning_rate=learning_rate, pre_tau=t_past))
            nengo.Connection(self.q, self.select)

            self.reward = nengo.Node(None, size_in=1)

            self.reward_array = nengo.Node(lambda t, x: x[:-1] * x[-1], size_in=3)
            nengo.Connection(self.reward, self.reward_array[-1])
            nengo.Connection(self.select, self.reward_array[:-1])

            self.error = nengo.Node(None, size_in=2)

            nengo.Connection(self.reward_array, self.error, synapse=t_past)
            nengo.Connection(self.q, self.error, synapse=t_now, transform=gamma)
            nengo.Connection(self.q, self.error, synapse=t_past, transform=-1)

            nengo.Connection(self.error, self.conn.learning_rule, transform=-1)

        import pdb
        pdb.set_trace()
        nengo.Connection(aigym[:-1], self.state)
        nengo.Connection(aigym[-1], self.reward)
        nengo.Connection(self.desired_action[0], aigym[0])
        nengo.Connection(self.desired_action[1], aigym[2])


class MountainCarTrial(pytry.NengoTrial):
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
            self.mc = OpenAIGym(actions=[0, 1, 2], name='MountainCar-v0', mean_solved=p.mean_solved, mean_cancel=-500, max_eps=p.max_eps, b_render=False)
            #self.ql = QLearn(aigym=self.mc, t_past=p.t_past, t_now=p.t_now, gamma=p.gamma, init_state = p.init_state)
            self.ql = QLearn1(aigym=self.mc, t_past=p.t_past, t_now=p.t_now, gamma=p.gamma, init_state=p.init_state, learning_rate=p.learning_rate)

        return model

    def evaluate(self, p, sim, plt):
        while not self.mc.reached_max_eps and not self.mc.cancel:
            sim.run(2)
        return dict(solved=self.mc.solved, episodes=self.mc.num_eps, last_hundred_rewards=self.mc.last_hundred_rewards)


model = nengo.Network(seed=2)
b_toy_cmd = False

with model:
    mc = OpenAIGym(actions=[0, 1, 2], name='MountainCar-v0')
    ql = QLearn1(aigym=mc)

    if b_toy_cmd:
        def input_func(t):
            result = [0] * 3
            index = int(np.random.rand(1, 1) * 2)

            if index > 0:
                index = 2

            result[index] = 1

            return result

        stim = nengo.Node(input_func)

        nengo.Connection(stim, mc)

if __name__ == '__main__':
    b_pytry = True
    if not b_pytry:
        sim = nengo.Simulator(model, progress_bar=False)
        while not mc.solved:
            sim.run(5)
    else:
        for t_past in [0.1]:
            for t_now in [0.005]:
                for gamma in [0.9]:
                    # for init_state in [[0,0,0], [0.5,0, 0.5]]:
                    for init_state in [[0, 0], [0.5, 0.5]]:
                        for learning_rate in [1e-3, 1e-4, 1e-5]:
                            MountainCarTrial().run(t_past=t_past, t_now=t_now, gamma=gamma, init_state=init_state, verbose=False)
