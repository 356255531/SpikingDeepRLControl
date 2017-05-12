#!/usr/bin/python
import matplotlib
matplotlib.backend = 'Qt4Agg'
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=4)
import os
import sys
import threading
import time

# import own modules
import agents
import goals
import q_networks
import replay_memory


# TODO: check if network folders exist; if not, make them
# TODO: look at soft updates; maybe test hard updates?
# TODO: look at MSE and convergence?


ARM_LENGTH_1 = 12.0
ARM_LENGTH_2 = 18.0
ANGULAR_ARM_VELOCITY = 1.0*np.pi/180.0

BATCH_SIZE = 64
EPSILON = 0.99
EPSILON_DECAY = 0.0005
GAMMA = 0.5
GOAL_THRESHOLD = 0.02
HEIGHT = 70
MAX_EPISODES = 500
MAX_STEPS = 500
MIN_SAMPLES = 4000
NUM_OF_ACTIONS = 4
NUM_OF_ACTORS = 8
NUM_OF_LEARNERS = 4
NUM_OF_PLOTS_X = 4
NUM_OF_PLOTS_Y = 2
NUM_OF_STATES = 6
STEPS_TO_SAVE_MODEL = 100
WIDTH = 70


class Actor(threading.Thread):
    def __init__(self, threadID, epsilon=EPSILON, max_steps=MAX_STEPS):
        threading.Thread.__init__(self)
        self.agent = None                       # place-holder for agent
        self.epsilon = epsilon 				    # (starting) exploration percentage
        self.goal = None 					    # placer-holder for goal
        self.MAX_STEPS = max_steps 				# maximal steps per episode
        self.THREAD_ID = threadID 				# thread id (integer)
        self.timestep = 0                       # timestep, used for exploration annealing

    def get_state(self):
    	# state is composed by agent + goal states
    	return np.hstack((self.agent.get_state(), self.goal.get_state()))

    def get_reward(self): 
        # penalize distance to goal
        agent_pos = self.agent.get_position()
        goal_pos = self.goal.get_position()
        distance = np.linalg.norm(agent_pos[:2] - goal_pos[:2])
        reward = -distance 
        return reward

    def episode_finished(self):
        agent_pos = self.agent.get_position()
        goal_pos = self.goal.get_position()
        distance = np.linalg.norm(agent_pos[:2] - goal_pos[:2])
        if distance < GOAL_THRESHOLD:
            return True # episode finished if agent already at goal
        else:
            return False

    def plot(self):
        # stepwise refreshing of plot
        ax[0,self.THREAD_ID].clear()
        
        # plotting of AGENT, GOAL and set AXIS LIMITS
        self.goal.plot(ax[0,self.THREAD_ID])
        self.agent.plot(ax[0,self.THREAD_ID])
        ax[0,self.THREAD_ID].set_xlim([-WIDTH/2, WIDTH/2])#[0,WIDTH])
        ax[0,self.THREAD_ID].set_ylim([-HEIGHT/2, HEIGHT/2])#[0,HEIGHT])

    def run(self):

        epsilon = self.epsilon
        for _ in range(MAX_EPISODES):#while True: 
            # init new episode
            plotting_lock.acquire()
            scene_id = np.random.choice([0,1,2,3])
            self.agent = agents.Arm(scene_id, angular_velocity_1=ANGULAR_ARM_VELOCITY, angular_velocity_2=ANGULAR_ARM_VELOCITY, arm_length_1=ARM_LENGTH_1, arm_length_2=ARM_LENGTH_2)
            self.goal = goals.Goal_Arm(scene_id, ARM_LENGTH_1, ARM_LENGTH_2)
            plotting_lock.release()

            self.timestep += 1
            
            for step in range(self.MAX_STEPS):
                # produce experience
                state = self.get_state()

                # get lock to synchronize threads
                networks_lock.acquire()
                q = networks.online_net.predict(state.reshape(1,NUM_OF_STATES), batch_size=1)
                networks_lock.release()

                random_number = np.random.uniform()
                if True: #random_number < epsilon: 
                    if random_number < epsilon/2:
                        action = np.random.randint(0, NUM_OF_ACTIONS) # choose random action
                    else:
                        # explore with guidance of inverse kinematics
                        try:
                            u = self.agent.get_control(self.goal.pos - self.agent.pos)
                            if np.argmax(np.abs(u)) == 0:
                                if u[0] < 0:
                                    action = 0
                                else:
                                    action = 1   
                            else:
                                if u[1] < 0:
                                    action = 2
                                else:
                                    action = 3
                        except:
                            action = np.random.randint(0, NUM_OF_ACTIONS) # choose random action if singularity
                else: 
                    action = np.argmax(q) # choose best action from Q(s,a)

                # take action, observe next state s'
                self.agent.set_action(action)
                self.agent.update()
                next_state = self.get_state()

                # observe reward
                reward = self.get_reward()

                # check if agent at goal
                terminal = self.episode_finished()

                # add exp sample to replay buffer
                replay_lock.acquire()
                replay.add_sample([state, action, reward, next_state, float(terminal)]) 
                replay_lock.release()

                # give console output and update plot
                console_lock.acquire()
                print '%3d | eps: %.2f | i: %3d | r: %.2f |' % (self.timestep, epsilon, step, reward), 'Q:', q
                console_lock.release()
                    
                # plot the scene
                plotting_lock.acquire()
                self.plot()
                plotting_lock.release()

                if terminal:
                    break # start new episode

            # explore less next time
            if epsilon > 0.1:
                epsilon = epsilon * 1.0/(1.0 + EPSILON_DECAY*self.timestep)

            # episodic refreshing of plot
            #plotting_lock.acquire()
            #ax[0,self.THREAD_ID].clear()
            #plotting_lock.release()

class Learner(threading.Thread):
    def __init__(self, threadID, gamma=GAMMA, min_samples=MIN_SAMPLES):
        threading.Thread.__init__(self)
        self.GAMMA = gamma
        self.MIN_SAMPLES = min_samples
        self.THREAD_ID = threadID

    def wait_for_replay_memory(self):
        while True:
            if replay.get_buffer_size() < self.MIN_SAMPLES:
                # no/too few experience buffered; wait a bit
                time.sleep(0.1)
            else:
                break
                 
    def run(self):
        # wait for enough experience samples
        self.wait_for_replay_memory()

        while True:
            for _ in range(STEPS_TO_SAVE_MODEL):
                # get lock to synchronize threads
                replay_lock.acquire()
                exp = replay.get_minibatch_samples(number_of_samples=BATCH_SIZE) # get exp samples from replay buffer
                replay_lock.release()

                # use experience to compute targets
                # TODO: find better method than np.array(*.tolist())
                states = np.array(exp[:,0].tolist(), np.float32)
                actions = np.array(exp[:,1].tolist()) # array+tolist not necessary
                rewards = np.array(exp[:,2].tolist(), np.float32) # array+tolist not necessary
                next_states = np.array(exp[:,3].tolist(), np.float32)
                terminals = np.array(exp[:,4].tolist()) # array+tolist not necessary

                # get lock to synchronize threads
                networks_lock.acquire()
                Q = networks.online_net.predict(states, batch_size=BATCH_SIZE) # get Q(s,a,theta) 
                newQ = networks.target_net.predict(next_states, batch_size=BATCH_SIZE) # get Q(s',a,theta^-) 
                networks_lock.release() 
                maxQ = np.max(newQ, axis=1) # get max_a Q(s',a,theta^-)
                    
                targets = np.copy(Q)
                targets[np.arange(BATCH_SIZE), actions[:]] = rewards + (1.0-terminals)*(self.GAMMA*maxQ) # target output
                # NOTE: (1.0-terminals) because if state is terminal, Q-learning target is defined only as reward without Q(s',a')
                
                # console output
                console_lock.acquire()
                print 'learning...'
                console_lock.release()

                # train online network on minibatch & apply soft updates on target network
                networks_lock.acquire()
                networks.online_net.train_on_batch(states, targets) 
                networks.do_soft_update()
                networks_lock.release() 

            # save online + target networks to disk
            networks_lock.acquire()
            networks.save_models()
            networks_lock.release() 


if __name__ == "__main__":
    # create GLOBAL thread-locks
    console_lock = threading.Lock()
    networks_lock = threading.Lock()
    replay_lock = threading.Lock()
    plotting_lock = threading.Lock()

    # create GLOBAL replay memory
    replay = replay_memory.ReplayMemory()

    # create GLOBAL Q-NETWORKS
    networks = q_networks.QNetworks(NUM_OF_ACTIONS, NUM_OF_STATES)

    # initialize GLOBAL plotting
    fig, ax = plt.subplots(NUM_OF_PLOTS_Y,NUM_OF_PLOTS_X)
    ax = ax.reshape(1, ax.shape[0]*ax.shape[1])
    plt.ion()

    # create threads
    threads = []
    threads.extend([Actor(i) for i in range(NUM_OF_ACTORS)])
    threads.extend([Learner(i) for i in range(NUM_OF_LEARNERS)])

    # set daemon, allowing Ctrl-C
    for i in range(len(threads)):
        threads[i].daemon = True

    # start new Threads
    [threads[i].start() for i in range(len(threads))]

    # show plot
    plt.show()
    while True:
        plotting_lock.acquire()
        fig.canvas.flush_events()
        plotting_lock.release()
        time.sleep(0.1)
