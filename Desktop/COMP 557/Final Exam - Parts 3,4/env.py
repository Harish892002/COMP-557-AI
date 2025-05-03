import random
from models import StateRescue
import numpy as np
from mdp import MDPBase, MDPRescueSimple, MDPRescueComplex


class EnvironmentBase:

    def __init__(self, **config):
        self.max_steps = 100
        self.current_step = 0
        self.mdp = None  # type: MDPBase
        self.cur_state = None  # type: StateRescue

        self.width = config['width']
        self.height = config['height']
        self.walls = config['walls']

    def reset(self):
        '''
        Reset the environment to the initial state.
        Return: the initial state of the environment.
        '''
        raise NotImplementedError

    def get_current_obj_locations(self):
        '''
        Return: the current object locations in the environment.
        '''
        raise NotImplementedError

    def get_current_state(self):
        '''
        Return: the current state of the environment.
        '''
        return self.cur_state

    def step(self, action):
        '''
        Take a step in the environment using the given action and 
            update the current state.
        Return: the next state and the reward.
        '''
        list_prob_nextstates = self.mdp.get_transition_distribution(
            self.cur_state, action)

        probs, nextstates = list(zip(*list_prob_nextstates))
        # sample next state
        next_state = random.choices(nextstates, weights=probs, k=1)[0]
        reward = self.mdp.get_reward(self.cur_state, action, next_state)

        self.cur_state = next_state
        self.current_step += 1

        return self.cur_state, reward

    def terminated(self):
        '''
        Check if the environment has terminated.
        The environment is terminated if all objects have been picked up
            or the maximum number of steps has been reached.
        '''
        if self.current_step >= self.max_steps:
            return True

        return self.mdp.is_terminal(self.cur_state)


class EnvRescueSimple(EnvironmentBase):

    def __init__(self, obj_loc, **config):
        super().__init__(**config)

        self.obj_loc = obj_loc
        self.init_loc = (3, 3)
        self.mdp = MDPRescueSimple(obj_loc, **config)

    def get_current_obj_locations(self):
        return [self.obj_loc]

    def reset(self):
        '''
        Reset the environment to the initial state.
        '''
        if self.init_loc is None:
            cur_loc = (random.randint(0, self.height - 1),
                       random.randint(0, self.width - 1))
        else:
            cur_loc = self.init_loc

        self.cur_state = StateRescue(self.width, self.height, self.walls,
                                     cur_loc)
        self.current_step = 0

        return self.cur_state


class EnvRescueComplex(EnvironmentBase):

    def __init__(self, n_objs, **config):
        super().__init__(**config)
        self.init_loc = (3, 3)
        self.mdp = MDPRescueComplex(n_objs, **config)

    def get_current_obj_locations(self):
        return self.cur_state.get_object_locations()

    def reset(self):
        '''
        Reset the environment to the initial state.
        '''
        if self.init_loc is None:
            cur_loc = (random.randint(0, self.height - 1),
                       random.randint(0, self.width - 1))
        else:
            cur_loc = self.init_loc

        sampled_obj_locs = random.sample(self.mdp.obj_locs, self.mdp.n_objs)

        obj_state = np.zeros((self.height, self.width), dtype=int)
        for r, c in sampled_obj_locs:
            obj_state[r, c] = 1

        self.cur_state = StateRescue(self.width, self.height, self.walls,
                                     cur_loc, obj_state)
        self.current_step = 0
        return self.cur_state
