import itertools
from models import Action, StateRescue, transition_model
import numpy as np


class MDPBase:

    def all_states(self):
        """
        Return: all possible states in the environment.
        """
        raise NotImplementedError

    def possible_actions(self, state):
        """
        Return: possible actions in the given state.
        """
        raise NotImplementedError

    def get_reward(self, state, action, next_state):
        """
        Get the reward for the given state, action, and next state.
        """
        raise NotImplementedError

    def is_terminal(self, state):
        """
        Check if the given state is a terminal state.
        """
        raise NotImplementedError

    def get_transition_distribution(self, state, action):
        '''
        Return: a list of tuples (prob, next_state) representing the
                transition distribution for the given state and action.
        '''
        raise NotImplementedError


class MDPRescueSimple(MDPBase):

    def __init__(self, obj_loc, **config):
        self.width = config['width']
        self.height = config['height']
        self.walls = config['walls']
        self.TERMINAL = StateRescue(self.width, self.height, self.walls,
                                    (-1, -1))

        self.__obj_loc = obj_loc

    def all_states(self):
        """
        Return: all possible states in the environment.
        """

        states = []
        for i in range(self.height):
            for j in range(self.width):
                agent_loc = (i, j)
                states.append(
                    StateRescue(self.width, self.height, self.walls,
                                agent_loc))
        states.append(self.TERMINAL)
        return states

    def possible_actions(self, state):
        """
        Return: possible actions in the given state.
        """
        actions = [
            Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY,
            Action.PICKUP
        ]
        return actions

    def get_reward(self, state: StateRescue, action: Action,
                   next_state: StateRescue):
        """
        Get the reward for the given state, action, and next state.
        """
        if self.is_terminal(state):
            return 0

        reward = -1
        coord = state.get_agent_location()
        if (coord[0] == self.__obj_loc[0] and coord[1] == self.__obj_loc[1]
                and action == Action.PICKUP):
            reward += 10

        return reward

    def is_terminal(self, state: StateRescue):
        """
        Check if the given state is a terminal state.
        """
        return state == self.TERMINAL

    def get_transition_distribution(self, state: StateRescue, action: Action):
        '''
        Return: a list of tuples (prob, next_state) representing the
                transition distribution for the given state and action.
        '''

        def can_pickup(state: StateRescue):
            coord = state.get_agent_location()
            return (coord[0] == self.__obj_loc[0]
                    and coord[1] == self.__obj_loc[1])

        return transition_model(self.width,
                                self.height,
                                self.walls,
                                state,
                                action,
                                self.TERMINAL,
                                cb_can_pickup=can_pickup)


class MDPRescueComplex(MDPBase):

    def __init__(self, n_objs, **config):
        self.width = config['width']
        self.height = config['height']
        self.walls = config['walls']
        self.TERMINAL = StateRescue(
            self.width, self.height, self.walls, (0, 0),
            np.zeros((self.height, self.width), dtype=int))

        self.obj_locs = [(i, j) for i in range(self.height)
                         for j in range(self.width)]
        self.n_objs = n_objs

    def all_states(self):
        """
        Return: all possible states in the environment.
        """
        objs = []
        for m in range(self.n_objs):
            objs = objs + list(itertools.combinations(self.obj_locs, m + 1))

        states = []
        for i in range(self.height):
            for j in range(self.width):
                agent_loc = (i, j)
                for locs in objs:
                    np_objs = np.zeros((self.height, self.width), dtype=int)
                    for r, c in locs:
                        np_objs[r, c] = 1
                    states.append(
                        StateRescue(self.width, self.height, self.walls,
                                    agent_loc, np_objs))
        states.append(
            StateRescue(self.width, self.height, self.walls, (0, 0),
                        np.zeros((self.height, self.width), dtype=int)))

        return states

    def possible_actions(self, state):
        '''
        Return: possible actions in the given state.
        '''
        actions = [
            Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY,
            Action.PICKUP
        ]
        return actions

    def get_reward(self, state: StateRescue, action: Action,
                   next_state: StateRescue):
        '''
        The reward is -1 for each step taken, and +10 for each object picked up.
        '''

        if self.is_terminal(state):
            return 0

        reward = -1
        n_objs = int(np.sum(state.obj_state))
        n_objs_next = int(np.sum(next_state.obj_state))

        if n_objs > n_objs_next:
            reward += 10 * (n_objs - n_objs_next)

        return reward

    def is_terminal(self, state: StateRescue):
        """
        Check if the given state is a terminal state.
        """
        return state == self.TERMINAL

    def get_transition_distribution(self, state: StateRescue, action: Action):
        '''
        Return: a list of tuples (prob, next_state) representing the
                transition distribution for the given state and action.
        '''

        def can_pickup(state: StateRescue):
            coord = state.get_agent_location()
            return state.obj_state[coord] != 0

        return transition_model(self.width,
                                self.height,
                                self.walls,
                                state,
                                action,
                                self.TERMINAL,
                                cb_can_pickup=can_pickup)
