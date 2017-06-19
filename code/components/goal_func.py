import numpy as np


class Goal(object):
    """
    Note, dependency here needed,
    1) state_action_space: transform the degree to state

    Member function:
        constructor(goal_coor, state_action_space)

        if_goal_state(state)

        get_goal_state() """

    def __init__(self, goal_coor, state_action_space):
        """
        args:
               goal_coor, numpy array, goal coordinate
               state_action_space, object """
        super(Goal, self).__init__()
        self._goal_coor = goal_coor
        self._goal_state = state_action_space.degree_to_state(goal_coor)

    def if_goal_state(self, state):
        """
        state:
               state, numpy array

        return:
                bool, if the given state is goal state """
        return np.array_equal(state, self._goal_state)

    def get_goal_state(self):
        """
        return:
                numpy array, goal state """
        return self._goal_state


def main():
    goal = Goal(np.array([10, 0]))
    print goal.get_goal_state()
    print goal.if_goal_state(np.array([10.0, 0]))


if __name__ == '__main__':
    main()
