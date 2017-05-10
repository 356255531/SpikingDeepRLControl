__auther__ = "zhiwei"
from agent import VirtualArm, RobotArm


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
                 action_space,
                 state_space,
                 reward_func,
                 goal_func,
                 if_emulator=True):
        super(RobotArmEnv, self).__init__()
        # Define dependent components
        self._action_space = action_space
        self._state_space = state_space
        self._reward_func = reward_func
        self._goal_func = goal_func

        # Define agent
        if if_emulator:
            self._arm = VirtualArm()
        else:
            self._arm = RobotArm()

    def init(self):
        # Init the local variables
        self._arm.init()
        arm_readout = self._arm.read()
        self._state = self._state_space.get_state(arm_readout)
        self._done = False

        return self._state

    def step(self, action):
        if not self._action_space.if_legal(action):
            raise ValueError("Action illegal")

        if self._done:
            raise ValueError("Episode ended, please reinitialize.")

        self._prev_state = self._state
        self._state = self._perform_action(action)

        return tuple(self._state, self._reward.evlt(self._prev_state, self._state), self._done)

    def _perform_action(self, action):
        arm_input = self._action_space.get_arm_input(action)
        self._arm.perform_action(arm_input)

        arm_readout = self._arm.read()
        state = self._state_space.get_state(arm_readout)

        if self._goal_func(state):
            self._done = True
        return state
