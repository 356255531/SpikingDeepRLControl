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
from collections import deque

# import own modules
import agents
import goals
import q_networks


ARM_LENGTH_1 = 12.0
ARM_LENGTH_2 = 18.0
ANGULAR_ARM_VELOCITY = 1.0*np.pi/180.0

GOAL_THRESHOLD = 0.02
HEIGHT = 70
MAX_STEPS = 500
NUM_OF_ACTIONS = 4
NUM_OF_ACTORS = 1
NUM_OF_PLOTS_X = 2
NUM_OF_PLOTS_Y = 1
NUM_OF_STATES = 6
WIDTH = 70


class Actor(threading.Thread):
    def __init__(self, threadID, goal_threshold=GOAL_THRESHOLD, max_steps=MAX_STEPS):
        threading.Thread.__init__(self)
        self.agent = None                       # place-holder for agent
        self.goal = None 					    # placer-holder for goal
        self.GOAL_THRESHOLD = goal_threshold 	# desired distance to goal; episode is finished early if threshold is achieved
        self.MAX_STEPS = max_steps 				# maximal steps per episode
        self.THREAD_ID = threadID 				# thread id (integer)
        self.path = deque([], maxlen=500)

    def get_state(self):
    	# state is composed by agent + goal states
    	return np.hstack((self.agent.get_state(), self.goal.get_state()))

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

        for point in self.path:
            ax[0,self.THREAD_ID].plot(point[0],point[1],'co')

    def run(self):
        while True: 
            # init new episode
            plotting_lock.acquire()
            scene_id = np.random.choice([0,1,2,3])
            self.agent = agents.Arm(scene_id, angular_velocity_1=ANGULAR_ARM_VELOCITY, angular_velocity_2=ANGULAR_ARM_VELOCITY, arm_length_1=ARM_LENGTH_1, arm_length_2=ARM_LENGTH_2)
            self.goal = goals.Goal_Arm(scene_id, ARM_LENGTH_1, ARM_LENGTH_2)
            plotting_lock.release()

            for step in range(self.MAX_STEPS):
                # produce experience
                state = self.get_state()
                self.path.append(self.agent.get_end_effector_position())

                # get lock to synchronize threads
                networks_lock.acquire()
                q = networks.online_net.predict(state.reshape(1,NUM_OF_STATES), batch_size=1)
                networks_lock.release()
                action = np.argmax(q) # choose best action from Q(s,a)

                # take action, observe next state s'
                self.agent.set_action(action)
                self.agent.update()
                next_state = self.get_state()

                # check if agent at goal
                terminal = self.episode_finished()
                    
                # plot the scene
                plotting_lock.acquire()
                self.plot()
                plotting_lock.release()

                if terminal:
                    break # start new episode

            # episodic refreshing of plot
            #plotting_lock.acquire()
            #ax[0,self.THREAD_ID].clear()
            #plotting_lock.release()


if __name__ == "__main__":
    # create GLOBAL thread-locks
    console_lock = threading.Lock()
    networks_lock = threading.Lock()
    plotting_lock = threading.Lock()

    # create GLOBAL Q-NETWORKS
    networks = q_networks.QNetworks(NUM_OF_ACTIONS, NUM_OF_STATES)

    # initialize GLOBAL plotting
    fig, ax = plt.subplots(NUM_OF_PLOTS_Y,NUM_OF_PLOTS_X)
    ax = ax.reshape(1, ax.shape[0]*ax.shape[1])
    plt.ion()

    # create threads
    threads = []
    threads.extend([Actor(i) for i in range(NUM_OF_ACTORS)])

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