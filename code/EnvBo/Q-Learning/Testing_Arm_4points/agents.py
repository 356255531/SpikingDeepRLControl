#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np


# ARM PARAMETERS
ANGULAR_ARM_VELOCITY = 1.0/180.0*np.pi
ARM_LENGTH_1 = 2.0
ARM_LENGTH_2 = 3.0
SCENARIOS = [(0,0),(0,30),(35,45),(0,150)]
# TODO: extend actions to all combinations, i.e. instead of 4 actions, all 3*3=9 actions (if too much time)

class Arm:
    def __init__(self, scene_id, angular_velocity_1=ANGULAR_ARM_VELOCITY, angular_velocity_2=ANGULAR_ARM_VELOCITY, arm_length_1=ARM_LENGTH_1, arm_length_2=ARM_LENGTH_2):
        self.base_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.ctrl = np.array([0.0, 0.0])
        self.theta = np.pi*np.array([SCENARIOS[scene_id][0],SCENARIOS[scene_id][1]], dtype=np.float32)/180 #[2.0*np.pi*np.random.randint(0,359)/360.0, 2.0*np.pi*np.random.randint(0,359)/360.0])
        self.vel = np.array([0.0, 0.0])

        self.ANGULAR_VELOCITY_1 = angular_velocity_1
        self.ANGULAR_VELOCITY_2 = angular_velocity_2
        self.ARM_LENGTH_1 = arm_length_1
        self.ARM_LENGTH_2 = arm_length_2
        
        self.pos = self.get_end_effector_position()
        
    def get_end_effector_position(self):
        pos = np.array([0.0, 0.0])
        pos[0] = self.base_pos[0] + self.ARM_LENGTH_1*np.cos(self.theta[0]) + self.ARM_LENGTH_2*np.cos(self.theta[0]+self.theta[1])
        pos[1] = self.base_pos[1] + self.ARM_LENGTH_1*np.sin(self.theta[0]) + self.ARM_LENGTH_2*np.sin(self.theta[0]+self.theta[1])
        return pos

    def get_Jacobian(self):
        return np.array([
                    [-np.sin(self.theta[0])*self.ARM_LENGTH_1 - self.ARM_LENGTH_2*np.sin(self.theta[0]+self.theta[1]), 
                    -self.ARM_LENGTH_2*np.sin(self.theta[0]+self.theta[1])], 

                    [np.cos(self.theta[0])*self.ARM_LENGTH_1 + self.ARM_LENGTH_2*np.cos(self.theta[0]+self.theta[1]), 
                    self.ARM_LENGTH_2*np.cos(self.theta[0]+self.theta[1])]
                ])

    def get_control(self, distance):
        J = self.get_Jacobian()
        u = np.dot(np.linalg.inv(J), distance)
        return u

    def get_position(self):
        normalized_pos = self.pos/(self.ARM_LENGTH_1+self.ARM_LENGTH_2)
        return np.hstack(normalized_pos)

    def get_state(self):
        normalized_pos = self.pos/(self.ARM_LENGTH_1+self.ARM_LENGTH_2)
        normalized_theta = (self.theta - np.pi)/np.pi
        return np.hstack((normalized_pos, normalized_theta))

    def plot(self, ax):
        # middle joint position
        pos = [ self.base_pos[0] + self.ARM_LENGTH_1*np.cos(self.theta[0]),
                self.base_pos[1] + self.ARM_LENGTH_1*np.sin(self.theta[0])]

        # TO DO: show velocity on arms
        linewidth = 5
        markersize = 10
        vel_factor = 100

        # arm links
        h_arm1 = ax.plot(   [self.base_pos[0], pos[0]], 
                            [self.base_pos[1], pos[1]], 
                            'k', linewidth=linewidth)
        h_arm2 = ax.plot(   [pos[0], self.pos[0]], 
                            [pos[1], self.pos[1]], 
                            'k', linewidth=linewidth)

        # base, middle and end-effector joints
        h_base = ax.plot(   self.base_pos[0], self.base_pos[1],
                            'ro', markersize=markersize, markeredgewidth=linewidth)
        h_middle = ax.plot(  pos[0], pos[1],
                            'ro', markersize=markersize, markeredgewidth=linewidth)
        h_end = ax.plot(    self.pos[0], self.pos[1],
                            'ro', markersize=markersize, markeredgewidth=linewidth)

        # velocity indicators
        h_vel1 = ax.plot(   [self.base_pos[0], self.base_pos[0]-vel_factor*self.vel[0]*np.sin(self.vel[0])], 
                            [self.base_pos[1], self.base_pos[1]+vel_factor*self.vel[0]*np.cos(self.vel[0])], 
                            'b', linewidth=linewidth/2)

        h_vel2 = ax.plot(   [pos[0], pos[0]-vel_factor*self.vel[1]*np.sin(self.vel[1])], 
                            [pos[1], pos[1]+vel_factor*self.vel[1]*np.cos(self.vel[1])], 
                            'b', linewidth=linewidth/2)

    def set_action(self, action):
        self.ctrl = np.array([0.0, 0.0]) # reset control
        if action==0:
            self.ctrl[0] = -self.ANGULAR_VELOCITY_1
        elif action==1:
            self.ctrl[0] = self.ANGULAR_VELOCITY_1
        elif action==2:
            self.ctrl[1] = -self.ANGULAR_VELOCITY_2
        elif action==3:
            self.ctrl[1] = self.ANGULAR_VELOCITY_2
                
    def update(self):
        # update (angular) velocities
        self.vel = self.ctrl
        
        # update positions
        self.theta += self.vel

        #re-map positions to environment
        self.theta = self.theta % (2.0*np.pi)

        # update end-effector position
        self.pos = self.get_end_effector_position()                


#agent = Arm(arm_length_1=ARM_LENGTH_1, arm_length_2=ARM_LENGTH_2)
#fig,ax = plt.subplots(1,1)
#agent.plot(ax)
#print agent.get_state()

#agent.set_action(1)
#agent.update()

#print agent.get_state()
#agent.plot(ax)

#plt.show()