__auther__ = "Zhiwei"
import numpy as np


class StateActionSpace_RobotArm(object):
    """
    Robot arm state action space class

    Member function:
            degree_to_state(arm_readout_in_degree)

            action_to_arm_input(action)

    Instance:
        _resolution, numpy array, used to discretize robot arm readout to state """

    def __init__(
            self,
            resolution=np.array([10]),
    ):
        super(StateActionSpace_RobotArm, self).__init__()
        self._resolution = resolution

    def degree_to_state(self, arm_readout_in_degree):
        """
        args:
            arm_readout_in_degree, numpy array

        return:
            artificial state corresponds to the robot arm readout in degree """
        state = []
        for observation_degree, resolution in zip(arm_readout_in_degree, self._resolution):
            state.append(-1 + 2.0 * (observation_degree // resolution) / (360 / resolution))
        return np.array(state)

    def action_to_arm_input(self, action):
        arm_input = []
        for idx, single_action in enumerate(action):
            if 0 == single_action:
                arm_input.append(-self._resolution[idx])
            if 1 == single_action:
                arm_input.append(0)
            if 2 == single_action:
                arm_input.append(self._resolution[idx])
        return np.array(arm_input)


def main():
    state_action_space = StateActionSpace_RobotArm([10, 20])
    print state_action_space.degree_to_state(np.array([360, 90]))
    print state_action_space.action_to_arm_input(np.array([2, 1]))


if __name__ == '__main__':
    main()
