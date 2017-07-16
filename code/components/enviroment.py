__auther__ = "zhiwei"
from agents import VirtualArm, RobotArm
from state_action_space import StateActionSpace_RobotArm
from reward_func import Reward
from goal_func import Goal

import numpy as np
import random as rd


class RobotArmEnv(object):
    """
        DRL Simulator Emulator class of NST omnibot

        Member function:
            constructor(state_action_space, reward_func, goal_func, if_simulator, if_visual, dim)

            init_game():
                            random init the robot arm

            step(action)

        Instance:
            _state_action_space
            _reward_func
            _goal_func
            _if_visual
            _state
            _done: if reach the goal state
            _previous_state """

    def __init__(self,
                 state_action_space,
                 reward_func,
                 goal_func,
                 if_simulator=True,
                 if_visual=False,
                 dim=1):
        """
        note, there are several dependencies for this class
            1) state space: where output of robot and state transformation are defined.
            2) reward function: map (s, a, s') to reward.
            3) goal function: judge if the input state is the goal state.
        args:
            1) if_simulator, bool, if use emulator or real robot arm
            2) if_visual, bool, if do visualization
            3) dim, int, joint dimension (if use emulator)
        """
        super(RobotArmEnv, self).__init__()
        # Define dependent components
        self._state_action_space = state_action_space
        self._reward_func = reward_func
        self._goal_func = goal_func

        self._if_visual = if_visual
        self._dim = dim
        # Define agent
        if if_simulator:
            self._arm = VirtualArm(
                dim=self._dim,
                goal_coor=self._goal_func.return_goal_coor(),
                if_visual=self._if_visual
            )
        else:
            self._arm = RobotArm()

    def reset(self):
        """
        usage:
            random init the robot arm """
        self._arm.init()
        current_end_coor = self._arm.read_end_coor()
        while self._goal_func.if_goal_coor(current_end_coor):
            self._arm.init()
            current_end_coor = self._arm.read_end_coor()

        arm_readout = self._arm.read_joint_degree()
        self._state = self._state_action_space.degree_to_state(arm_readout)
        self._done = False

        return self._state

    def step(self, action):
        """
        args:
            action, int

        usage:
            step the enviroment forward with given action,
            update the state, previous state, reward and so on. """
        if self._done:
            raise ValueError("Episode ended, please reinitialize.")

        self._prev_state = self._state

        if action[0] != 0 and action[1] != 0:
            raise ValueError
        self._state = self._perform_action(action)

        current_end_coor = self._arm.read_end_coor()
        return \
            self._state, \
            self._reward_func.evlt(
                current_end_coor,
                self._goal_func.return_goal_coor()
            ), \
            self._done

    def _perform_action(self, action):
        arm_input = self._state_action_space.action_to_arm_input(action)
        self._arm.perform_action(arm_input)
        arm_readout = self._arm.read_joint_degree()
        state = self._state_action_space.degree_to_state(arm_readout)

        current_end_coor = self._arm.read_end_coor()
        if self._goal_func.if_goal_coor(current_end_coor):
            self._done = True
        return state

    def get_jacobi_action(self):
        J = self._arm.get_jocobian()
        distance = self._goal_func.return_goal_coor() - self._arm.read_end_coor()
        action_in_degree = np.dot(np.linalg.pinv(J), distance)

        action_idx = np.argmax(np.abs(action_in_degree))

        action = np.zeros(2)

        if action_in_degree[action_idx] > 0:
            action[action_idx] = 0
        else:
            action[action_idx] = 2

        return action

    def close(self):
        pass


def main():
    resolution_in_degree = 10 * np.ones(2)  # Discretization Resolution in Degree
    state_action_space = StateActionSpace_RobotArm(resolution_in_degree)  # Encode the joint to state

    reward_func = Reward()  # The rule of reward function
    goal_func = Goal((-3, 0))
    env = RobotArmEnv(
        state_action_space,
        reward_func,
        goal_func,
        if_simulator=True,
        if_visual=True,
        dim=2
    )
    env.reset()
    done = False
    while not done:
        action = env.get_jacobi_action()
        state, reward, done = env.step(action)
        print action
        if done:
            break


if __name__ == '__main__':
    main()
