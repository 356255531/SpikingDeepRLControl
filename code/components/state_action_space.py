__auther__ = "Zhiwei"
import numpy as np


class StateActionSpace_RobotArm(object):
    """docstring for StateActionSpace_RobotArm"""

    def __init__(
            self,
            resolution=np.array([10]),
    ):
        super(StateActionSpace_RobotArm, self).__init__()
        self._resolution = resolution

    def degree_to_state(self, observation):
        state = []
        for observation_degree, resolution in zip(observation, self._resolution):
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
