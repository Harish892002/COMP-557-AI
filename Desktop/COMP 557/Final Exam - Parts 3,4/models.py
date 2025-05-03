import numpy as np

MAP_V1 = {
    "name":
    "v1",
    "width":
    7,
    "height":
    7,
    "walls":
    tuple([(0, 3, 'v'), (1, 5, 'v'), (2, 2, 'v'), (2, 2, 'h'), (2, 5, 'h'),
           (3, 0, 'h'), (4, 5, 'v'), (4, 6, 'h'), (5, 1, 'h'), (5, 2, 'v'),
           (5, 4, 'h'), (6, 4, 'v')])
}


class Action:
    UP = "Up"
    DOWN = "Down"
    LEFT = "Left"
    RIGHT = "Right"
    STAY = "Stay"
    PICKUP = "Pickup"


class StateRescue():

    def __init__(self, width, height, walls, agent_loc, obj_state=None):
        self.agent_loc = agent_loc  # (x, y)
        self.obj_state = None
        if obj_state is not None:
            self.obj_state = obj_state.astype(int)  # n by n array of 0s and 1s

        self._height = height
        self._width = width
        self._walls = walls

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def walls(self):
        return self._walls

    def get_object_locations(self):
        'return the locations of objects'
        if self.obj_state is None:
            return None

        indices = np.argwhere(self.obj_state == 1).tolist()
        locs = [tuple(idx) for idx in indices]
        return locs

    def get_agent_location(self):
        return self.agent_loc

    def __repr__(self):
        return f"StateRescue(agent_loc={self.get_agent_location()}, obj_state={self.get_object_locations()})"

    def __hash__(self):
        if self.obj_state is None:
            return hash((self.agent_loc[0], self.agent_loc[1]))

        # assume height and width are less than 10
        height = self.obj_state.shape[0]
        width = self.obj_state.shape[1]

        text = str(height) + str(width)
        text += str(self.agent_loc[0]) + str(self.agent_loc[1])

        max_decimal = 2**width - 1
        pad_width = len(str(max_decimal))  # Number of digits needed

        # Convert each row of the 2D array into a decimal number with padding
        for row in self.obj_state:
            binary_str = ''.join(map(str, row))
            decimal_val = int(binary_str, 2)
            text += str(decimal_val).zfill(pad_width)  # Pad with leading zeros

        return hash(text)

    def __eq__(self, other):
        if self.obj_state is None:
            return (isinstance(other, StateRescue)
                    and self.agent_loc == other.agent_loc
                    and other.obj_state is None)
        else:
            return (isinstance(other, StateRescue)
                    and self.agent_loc == other.agent_loc
                    and np.array_equal(self.obj_state, other.obj_state))


def get_reachable_neighbors(coord, walls, width, height):
    '''
    Return: neighboring cells of the given coordinate that are reachable
            (i.e., not blocked by walls).
    '''
    neighbors = []

    # check if the cell above is accessible
    if (coord[0] > 0 and (coord[0], coord[1], 'h') not in walls):
        neighbors.append((coord[0] - 1, coord[1]))
    # check if the cell below is accessible
    if (coord[0] < height - 1 and (coord[0] + 1, coord[1], 'h') not in walls):
        neighbors.append((coord[0] + 1, coord[1]))
    # check if the cell to the left is accessible
    if (coord[1] > 0 and (coord[0], coord[1], 'v') not in walls):
        neighbors.append((coord[0], coord[1] - 1))
    # check if the cell to the right is accessible
    if (coord[1] < width - 1 and (coord[0], coord[1] + 1, 'v') not in walls):
        neighbors.append((coord[0], coord[1] + 1))

    return neighbors


def transition_model(width, height, walls, cur_state: StateRescue, action,
                     terminal_state, cb_can_pickup):
    cur_loc = cur_state.agent_loc
    obj_state = cur_state.obj_state
    if terminal_state == cur_state:
        return [(1.0, cur_state)]

    # check reachable locations
    reachable_locs = get_reachable_neighbors(cur_loc, walls, width, height)
    # check reachable locations
    left = (cur_loc[0], cur_loc[1] - 1)
    right = (cur_loc[0], cur_loc[1] + 1)
    up = (cur_loc[0] - 1, cur_loc[1])
    down = (cur_loc[0] + 1, cur_loc[1])

    P_DESIRED = 0.95
    P_ERROR = 1.0 - P_DESIRED
    P_PICKUP = 1.0

    next_state_w_p = []
    if action == Action.STAY:
        n_reachable_locs = len(reachable_locs)
        if n_reachable_locs == 0:
            next_state_w_p.append((1.0, cur_state))
        else:
            next_state_w_p.append((P_DESIRED, cur_state))
            for i in range(n_reachable_locs):
                next_state_w_p.append(
                    (P_ERROR / n_reachable_locs,
                     StateRescue(width, height, walls, reachable_locs[i],
                                 obj_state)))
    elif action == Action.PICKUP:
        if cb_can_pickup(cur_state):
            if obj_state is None:
                next_state_w_p.append((P_PICKUP, terminal_state))
                next_state_w_p.append((1 - P_PICKUP, cur_state))
            else:
                # pickup object
                new_obj_state = np.array(obj_state, dtype=int)
                new_obj_state[tuple(cur_loc)] = 0
                if np.sum(new_obj_state) == 0:
                    new_state = terminal_state
                else:
                    new_state = StateRescue(width, height, walls, cur_loc,
                                            new_obj_state)

                next_state_w_p.append((P_PICKUP, new_state))
                next_state_w_p.append((1 - P_PICKUP, cur_state))
        else:
            # cannot pickup object
            next_state_w_p.append((1.0, cur_state))

    else:
        if action == Action.UP:
            desired_loc = up
            rand_locs = [left, right]
        elif action == Action.DOWN:
            desired_loc = down
            rand_locs = [left, right]
        elif action == Action.LEFT:
            desired_loc = left
            rand_locs = [down, up]
        elif action == Action.RIGHT:
            desired_loc = right
            rand_locs = [down, up]
        else:
            raise ValueError("Invalid action")

        next_loc = desired_loc if desired_loc in reachable_locs else cur_loc
        rand_loc1 = rand_locs[0] if rand_locs[0] in reachable_locs else cur_loc
        rand_loc2 = rand_locs[1] if rand_locs[1] in reachable_locs else cur_loc

        next_state_w_p.append(
            (P_DESIRED, StateRescue(width, height, walls, next_loc,
                                    obj_state)))
        next_state_w_p.append((P_ERROR / 2,
                               StateRescue(width, height, walls, rand_loc1,
                                           obj_state)))
        next_state_w_p.append((P_ERROR / 2,
                               StateRescue(width, height, walls, rand_loc2,
                                           obj_state)))

    # clean probabilities
    dict_next_states = {}
    for prob, ns in next_state_w_p:
        if prob > 0:
            dict_next_states[ns] = dict_next_states.get(ns, 0) + prob

    # list_p_with_state
    output = []
    for key, value in dict_next_states.items():
        output.append((value, key))
    return output
