import numpy as np
from goal_func import Goal


class Reward(object):
    """
    Reward function class

    Member function:
        evtl(previous_state, action, state, goal_state) """

    def __init__(self):
        super(Reward, self).__init__()

    def evlt(self, current_coor, goal_coor):
        """
        args:
            previous_state, numpy array
            action, numpy array
            state, numpy array
            goal state, numpy array

        return:
            reward accordingto the setting above """
        if np.linalg.norm(current_coor-  goal_coor) < 0.52:
            return 1
        return -np.linalg.norm(goal_coor - current_coor)
