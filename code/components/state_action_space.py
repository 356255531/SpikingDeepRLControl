__auther__ = "Zhiwei"
import numpy as np


class StateActionSpace_RobotArm(object):
    """docstring for StateActionSpace_RobotArm"""

    def __init__(
            self,
            dim=1,
            resolution=np.array([10]),
            action_unit_in_degree=np.array([3])
    ):
        super(StateActionSpace_RobotArm, self).__init__()
        self._dim = dim
        self._resolution = resolution
        self._action_unit_in_degree = action_unit_in_degree

    def degree_to_state(self, observation):
        observation = np.array(observation)
        state = []
        for observation_degree, resolution in zip(observation, self._resolution):
            state.append(observation_degree // resolution)
        return tuple(state)

    def get_arm_input(self, action):
        arm_input = []
        for single_action in action:
            if 0 == single_action:
                arm_input.append(-10)
            if 1 == single_action:
                arm_input.append(0)
            if 2 == single_action:
                arm_input.append(10)
        return arm_input


def main():
    state_action_space = StateActionSpace_RobotArm()
    print state_action_space.degree_to_state(np.array([20]))


if __name__ == '__main__':
    main()
