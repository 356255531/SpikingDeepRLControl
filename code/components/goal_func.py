import numpy as np


class Goal(object):
    """
    Note, dependency here needed,
    1) state_action_space: transform the degree to state

    Member function:
        constructor(goal_coor, state_action_space)

        if_goal_coor(state)

        get_goal_state() """

    def __init__(self, goal_coor):
        """
        args:
               goal_coor, numpy array, goal coordinate
               state_action_space, object """
        super(Goal, self).__init__()
        self._goal_coor = goal_coor

    def if_goal_coor(self, current_coor):
        """
        state:
               state, numpy array

        return:
                bool, if the given state is goal state """
        return np.linalg.norm(current_coor - self._goal_coor) < 1

    def return_goal_coor(self):
        return self._goal_coor


def main():
    goal = Goal(np.array([10, 0]))
    print goal.get_goal_state()
    print goal.if_goal_coor(np.array([10.0, 0]))


if __name__ == '__main__':
    main()
