__auther__ = "zhiwei"
import operator


class VirtualArm(object):
    """docstring for VirtualArm"""

    def __init__(self, free_degree, upper_bound, lower_bound, action_delay=0):
        super(VirtualArm, self).__init__()
        if not isinstance(free_degree, int) or free_degree < 1:
            raise ValueError("Free degree illegal")
        if free_degree != len(upper_bound) or free_degree != len(lower_bound):
            raise ValueError("Bound dimension error")
        self._free_degree = free_degree
        self._upper_bound = upper_bound
        self._lower_bound = lower_bound
        self._action_delay = action_delay

    def init(self):
        self._position = tuple([0.0 for i in xrange(self._free_degree)])

    def perform_action(self, arm_input):
        if not self._check_input(arm_input):
            raise ValueError("arm input illegal")

        position = tuple(map(operator.add, self._position, arm_input))

        if any(joint > self._upper_bound[idx] or joint < self.lower_bound[idx] for idx, joint in enumerate(position)):
            raise ValueError("Arm input would lead to joint overflow")

        self._position = position

    def read(self):
        return self._position


class RobotArm(object):
    """docstring for RobotArm"""

    def __init__(self, arg):
        super(RobotArm, self).__init__()
        self.arg = arg
