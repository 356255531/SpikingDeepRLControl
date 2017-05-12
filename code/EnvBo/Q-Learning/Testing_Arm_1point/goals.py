#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np


MARKERSIZE = 15
LINEWIDTH = 3

    

class Goal_Arm:
    def __init__(self, arm_length1, arm_length2):
        self.ARM_LENGTH_1 = arm_length1
        self.ARM_LENGTH_2 = arm_length2
        angle1 = 100*2.0*np.pi/360.0 #float(np.random.randint(0,359))*2.0*np.pi/360.0
        angle2 = 40*2.0*np.pi/360.0 #float(np.random.randint(0,359))*2.0*np.pi/360.0
        x = arm_length1*np.cos(angle1) + arm_length2*np.cos(angle1+angle2)
        y = arm_length1*np.sin(angle1) + arm_length2*np.sin(angle1+angle2)
        self.pos = np.array([x, y])        

    def plot(self, ax):
        ax.plot(    self.pos[0], 
                    self.pos[1], 
                    'bo', markersize=MARKERSIZE, markeredgewidth=LINEWIDTH)

    def get_position(self):
        # goal position is normalized 2D [x,y] (positions are between -1 and 1)
        normalized_pos = self.pos / (self.ARM_LENGTH_1+self.ARM_LENGTH_2)
        return normalized_pos

    def get_state(self):
        # goal state is only goal position
        normalized_pos = self.pos / (self.ARM_LENGTH_1+self.ARM_LENGTH_2)
        return normalized_pos