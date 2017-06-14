import numpy as np


class Goal(object):
    """docstring for Goal"""

    def __init__(self, goal_coor, state_action_space):
        super(Goal, self).__init__()
        self._goal_coor = goal_coor
        self._goal_state = state_action_space.degree_to_state(goal_coor)

    def if_goal_state(self, state):
        return np.array_equal(state, self._goal_state)

    def get_goal_state(self):
        return self._goal_state


def main():
    goal = Goal(np.array([10, 0]))
    print goal.get_goal_state()
    print goal.if_goal_state(np.array([10.0, 0]))


if __name__ == '__main__':
    main()
