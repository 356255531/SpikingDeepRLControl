__auther__ = "zhiwei"
from agents import VirtualArm, RobotArm
from state_action_space import StateActionSpace_RobotArm
from reward_func import Reward
from goal_func import Goal

import numpy as np
import random as rd


class RobotArmEnv(object):
    """ This is the docs for the class of NST robot arm emulator.

        Note:
            1. constructor denpendencies are
                1) state space: where the artificial state and its relationship
                    with Robotarm readout defined.
                2) action state: where the artifical action and its relationship
                    with robotarm input defined.
                3) reward function: where the behavior of reward defined. It maps
                    (previous_state, current_state) -> single_step_reward.
                4) goal function: judge if the input state is the goal state.
                variables:
                5) if_emulator: decide if use a emulator or real robot arm.

        Usage:
            1. init() is to set the enviroment to init position and return the init
                state
            2. step(action) is to perform the given action ane return a tuple of
                (state, reward, if_done). Note, it will raise an error when the
                simulation is done.
                    """

    def __init__(self,
                 state_action_space,
                 reward_func,
                 goal_func,
                 if_emulator=True,
                 if_visual=False,
                 dim=1):
        super(RobotArmEnv, self).__init__()
        # Define dependent components
        self._state_action_space = state_action_space
        self._reward_func = reward_func
        self._goal_func = goal_func

        self._if_visual = if_visual
        # Define agent
        if if_emulator:
            self._arm = VirtualArm(
                dim=1,
                start_angular=np.zeros(1),
                goal_coor=(-3, 0),
                if_visual=self._if_visual
            )
        else:
            self._arm = RobotArm()

    def init_game(self):
        # init_game the local variables
        self._arm.init()
        arm_readout = self._arm.read()
        self._state = self._state_action_space.degree_to_state(arm_readout)
        self._done = False

        return self._state

    def step(self, action):
        if self._done:
            raise ValueError("Episode ended, please reinitialize.")

        self._prev_state = self._state

        self._state = self._perform_action(action)

        return \
            self._state, \
            self._reward_func.evlt(
                self._prev_state,
                action,
                self._state,
                self._goal_func.get_goal_state()
            ), \
            self._done

    def _perform_action(self, action):
        arm_input = self._state_action_space.action_to_arm_input(np.array([np.argmax(action)]))
        self._arm.perform_action(arm_input)
        arm_readout = self._arm.read()
        state = self._state_action_space.degree_to_state(arm_readout)

        if self._goal_func.if_goal_state(state):
            self._done = True
        return state


def main():
    state_action_space = StateActionSpace_RobotArm()
    reward = Reward()
    goal = Goal((3, 0), state_action_space)
    env = RobotArmEnv(state_action_space, reward, goal)
    env.init_game()
    done = False
    count = 0
    while not done:
        state, reward, done = env.step(np.array([rd.randint(-1, 1)]))
        print state, reward, done
        if done:
            break


if __name__ == '__main__':
    main()
