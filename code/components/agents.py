__auther__ = "zhiwei"
import numpy as np
import matplotlib.pyplot as plt

ARM_LENGTH_1 = 3.0
ARM_LENGTH_2 = 3.0
ARM_LENGTH_3 = 3.0
ARM_LENGTH = np.array([ARM_LENGTH_1, ARM_LENGTH_2, ARM_LENGTH_3])

PI = np.pi


class VirtualArm(object):
    """
    Agent class (multi-dim possible)

    Member function:
        constructor(dim, start_angular, goal_coor, if_visual)

        init(start_angular, goal_coor):

        perform_action(input_in_degree)

        read()
    Instance:
                _dim, numpy array
                _arm_len, numpy array
                _goal_coor, numpy array
                _if_visual, bool
    """

    def __init__(self,
                 dim=1,
                 start_angular=None,
                 goal_coor=np.array([-3, 0]),
                 if_visual=False,
                 ):
        """
        constructor(dim, start_angular, goal_coor, if_visual)
            arg:
                dim, int
                start_angular, numpy array
                goal_coor, numpy array
                if_visual, bool

            usage:
                    Create emulator class """
        super(VirtualArm, self).__init__()

        self._dim = dim
        self._arm_len = ARM_LENGTH[0: self._dim]

        self._goal_coor = goal_coor
        self._if_visual = if_visual

        if start_angular is None:
            start_angular = np.zeros(self._dim)
        self.init(start_angular)

    def _refresh_joint_coor(self):
        """
        init(start_angular, goal_coor):
            arg:
                    start_angular, numpy array
                    goal_coor, numpy
            usage:
                    Init the emulator with a start angular and goal state,
                    otherwise with random init joint degree """
        self._end_coor = np.array([[0.0, 0.0]])
        for angluar, arm_len in zip(self._arm_angulars_in_degree, self._arm_len):
            self._end_coor = np.append(
                self._end_coor,
                [[self._end_coor[-1][0] + arm_len * np.cos((angluar / 180.0) * np.pi),
                  self._end_coor[-1][1] + arm_len * np.sin((angluar / 180.0) * np.pi)]],
                axis=0
            )

    def init(self, start_angular=None, goal_coor=None):

        if goal_coor is not None:
            self._goal_coor = goal_coor

        if start_angular is None:
            self._arm_angulars_in_degree = np.random.randint(0, 36, size=[self._dim]) * 10
            # self._arm_angulars_in_degree = np.zeros(self._dim)
        else:
            self._arm_angulars_in_degree = start_angular

        self._refresh_joint_coor()

        if self._if_visual:
            try:
                self._visualize()
            except:
                import pdb
                pdb.set_trace()

    def perform_action(self, arm_input):
        """
        perform_action(input_in_degree):
            arg:
                    input_in_degree, numpy array

            usage:
                    give a rotation in degree """
        joint_accumulator = [arm_input[0]]
        for idx in xrange(1, len(arm_input)):
            joint_accumulator.append((joint_accumulator[-1] + arm_input[idx]) % 360)

        self._arm_angulars_in_degree = (self._arm_angulars_in_degree + np.array(joint_accumulator)) % 360

        self._refresh_joint_coor()

        if self._if_visual:
            self._visualize()

    def read_joint_degree(self):
        """
        read():
            return:
                    current all joints angular in degree, numpy array"""
        return self._arm_angulars_in_degree

    def read_end_coor(self):
        return self._end_coor[-1]

    def _visualize(self):

        linewidth = 1
        markersize = 3

        plt.gcf().clear()

        plt.axis([-10, 10, -10, 10])
        for i in xrange(self._dim):
            plt.plot(
                [self._end_coor[i][0], self._end_coor[i + 1][0]],
                [self._end_coor[i][1], self._end_coor[i + 1][1]],
                'k', linewidth=linewidth)

        for coor in self._end_coor:
            plt.plot(
                coor[0], coor[1],
                'ro', color='k',
                markersize=markersize, markeredgewidth=linewidth)

        plt.plot(self._goal_coor[0], self._goal_coor[1], 'ro', markersize=markersize, markeredgewidth=linewidth)
        plt.pause(0.01)

    def get_jocobian(self):
        if self._dim == 1:
            print "No need to use Jacobian control"
            return
        if self._dim == 2:
            Jacobian = np.array([
                [-np.sin(self._arm_angulars_in_degree[0] / 180) * ARM_LENGTH_1 - ARM_LENGTH_2 * np.sin(self._arm_angulars_in_degree[0] / 180 + self._arm_angulars_in_degree[1] / 180),
                 -ARM_LENGTH_2 * np.sin(self._arm_angulars_in_degree[0] / 180 + self._arm_angulars_in_degree[1] / 180)],

                [np.cos(self._arm_angulars_in_degree[0] / 180) * ARM_LENGTH_1 + ARM_LENGTH_2 * np.cos(self._arm_angulars_in_degree[0] / 180 + self._arm_angulars_in_degree[1] / 180),
                 ARM_LENGTH_2 * np.cos(self._arm_angulars_in_degree[0] / 180 + self._arm_angulars_in_degree[1] / 180)]
            ])
        return Jacobian


class RobotArm(object):
    """
        Class for the robot arm TODO"""

    def __init__(self, arg):
        super(RobotArm, self).__init__()
        self.arg = arg


def main():
    arm = VirtualArm(dim=2, if_visual=True)
    for x in xrange(1, 1000):
        current_coor = arm.read_end_coor()
        print "ha", current_coor
        arm.perform_action((10, 10))
        print "perform 10, 10, 10"


if __name__ == '__main__':
    main()
