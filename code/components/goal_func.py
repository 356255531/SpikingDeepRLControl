class Goal(object):
    """docstring for Goal"""

    def __init__(self, goal_state):
        super(Goal, self).__init__()
        self._goal_state = goal_state

    def if_goal(self, state):
        return state == self._goal_state

    def get_goal_state(self):
        return self._goal_state
