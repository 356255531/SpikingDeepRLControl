__auther__ = "zhiwei"
import numpy as np
import operator
import matplotlib.pyplot as plt

ARM_LENGTH_1 = 3.0
ARM_LENGTH_2 = 3.0
ARM_LENGTH_3 = 3.0
ARM_LENGTH = np.array([ARM_LENGTH_1, ARM_LENGTH_2, ARM_LENGTH_3])

PI = np.pi


class VirtualArm(object):
    """
    Member function:
        init(start_angular, goal_coor):
            arg:    start_angular, numpy array
                    goal_coor, numpy
        perform_action(input_in_degree):
            arg: input_in_degree, numpy array
            output: no output
        read():
            arg: no arg
            output: angular_of_all_joints, numpy array"""

    def __init__(self,
                 dim=1,
                 start_angular=np.zeros(1),
                 goal_coor=(-3, 0),
                 if_visual=True
                 ):
        super(VirtualArm, self).__init__()

        self._dim = dim
        self._arm_len = ARM_LENGTH[0: self._dim]

        self._goal_coor = goal_coor
        self._if_visual = if_visual
        self.init(start_angular)

    def _refresh_joint_coor(self):
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
            self._arm_angulars_in_degree = np.random.randint(360, size=[self._dim])
        else:
            self._arm_angulars_in_degree = start_angular

        self._refresh_joint_coor()

        if self._if_visual:
            self._visualize()

    def perform_action(self, arm_input):
        joint_accumulator = [arm_input[0]]
        for idx in xrange(1, len(arm_input)):
            joint_accumulator.append((joint_accumulator[-1] + arm_input[idx]) % 360)

        self._arm_angulars_in_degree = (self._arm_angulars_in_degree + np.array(joint_accumulator)) % 360

        self._refresh_joint_coor()

        if self._if_visual:
            self._visualize()

    def read(self):
        return self._arm_angulars_in_degree

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


class RobotArm(object):
    """
        Class for the robot arm TODO"""

    def __init__(self, arg):
        super(RobotArm, self).__init__()
        self.arg = arg


def main():
    arm = VirtualArm(dim=3,
                     start_angular=np.zeros(3),
                     goal_coor=(0, 3),
                     if_visual=True
                     )
    for x in xrange(1, 1000):
        arm.perform_action((10, 10, 10))
        print "perform 10, 10, 10"
        print arm.read()
        print arm._end_coor


if __name__ == '__main__':
    main()
