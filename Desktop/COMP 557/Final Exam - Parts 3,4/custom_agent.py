# agent_custom.py

from typing import List, Tuple, Dict
import models as mdl
import rl_utils as rlu
from env import EnvironmentBase
from collections import deque
from itertools import permutations
from models import get_reachable_neighbors
from heapq import heappush, heappop

# Slip probability in Rescue domains
P_DESIRED = 0.95

class CustomAgent(rlu.BaseAgent):
    """
    TSP-based, slip-aware route optimizer:
      1) Precompute adjacency for grid.
      2) On first call or after pickup/slip, solve TSP to order kit visits by BFS distances.
      3) For each leg, plan actual path via A*.
      4) Execute one-step actions along full planned path.
    """

    def __init__(self, env: EnvironmentBase):
        self.mdp = env.mdp
        self.walls = self.mdp.walls
        self.width = env.width
        self.height = env.height

        # Precompute neighbor map
        self.neighbors: Dict[Tuple[int,int], List[Tuple[int,int]]] = {
            (r, c): get_reachable_neighbors((r, c), self.walls, self.width, self.height)
            for r in range(self.height) for c in range(self.width)
        }

        # Planning state
        self._plan: List[Tuple[int,int]] = []
        self._plan_actions: List[mdl.Action] = []
        self._kits_set = frozenset()

    def get_best_action(self, state: mdl.StateRescue) -> mdl.Action:
        loc = state.get_agent_location()
        kits = state.get_object_locations() or []

        # 1) No kits left
        if not kits:
            return mdl.Action.STAY
        # 2) On kit cell -> pickup and reset plan
        if loc in kits:
            self._plan = []
            self._plan_actions = []
            self._kits_set = frozenset(kits)
            return mdl.Action.PICKUP

        kits_set = frozenset(kits)
        # 3) Replan if kits changed or plan exhausted or slip off
        if (kits_set != self._kits_set
            or not self._plan_actions
            or (self._plan and loc != self._plan[0])):
            self._kits_set = kits_set
            self._compute_full_plan(loc, kits)

        # 4) Pop next action
        if not self._plan_actions:
            return mdl.Action.STAY
        return self._plan_actions.pop(0)

    def _compute_full_plan(self, start: Tuple[int,int], kits: List[Tuple[int,int]]):
        # BFS distances from any cell to any cell via neighbor map
        def bfs_dist(origin):
            dmap = {origin: 0}
            dq = deque([origin])
            while dq:
                cur = dq.popleft()
                for nbr in self.neighbors[cur]:
                    if nbr not in dmap:
                        dmap[nbr] = dmap[cur] + 1
                        dq.append(nbr)
            return dmap

        # Precompute distances from start and each kit
        points = [start] + kits
        dist_maps = {p: bfs_dist(p) for p in points}

        # Solve TSP by brute-force on kits only
        best_order = None
        best_len = float('inf')
        for perm in permutations(kits):
            length = 0
            cur = start
            for k in perm:
                length += dist_maps[cur].get(k, float('inf'))
                cur = k
            # minimize raw length
            if length < best_len:
                best_len = length
                best_order = perm

        # Build waypoint sequence: start -> k1 -> k2 -> ...
        waypoints = [start] + list(best_order)

        # Plan each leg via A* and collect actions
        self._plan = []
        self._plan_actions = []
        for i in range(len(waypoints)-1):
            leg = self._a_star(waypoints[i], waypoints[i+1])
            self._plan.extend(leg)
        # translate coords to actions
        actions = []
        for idx in range(1, len(self._plan)):
            fr = self._plan[idx-1]
            to = self._plan[idx]
            dr, dc = to[0]-fr[0], to[1]-fr[1]
            if dr == -1: actions.append(mdl.Action.UP)
            elif dr == 1: actions.append(mdl.Action.DOWN)
            elif dc == -1: actions.append(mdl.Action.LEFT)
            elif dc == 1: actions.append(mdl.Action.RIGHT)
        self._plan_actions = actions

    def _a_star(self, start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
        # Weighted move cost under slip
        wcost = 1.0 / P_DESIRED
        # heuristic: Manhattan * wcost
        h = lambda x: (abs(x[0]-goal[0]) + abs(x[1]-goal[1])) * wcost

        open_h = []
        heappush(open_h, (h(start), 0.0, start, [start]))
        gscore = {start: 0.0}
        closed = set()

        while open_h:
            _, g, cur, path = heappop(open_h)
            if cur == goal:
                return path
            if cur in closed:
                continue
            closed.add(cur)
            for nbr in self.neighbors[cur]:
                ng = g + wcost
                if ng < gscore.get(nbr, float('inf')):
                    gscore[nbr] = ng
                    f = ng + h(nbr)
                    heappush(open_h, (f, ng, nbr, path + [nbr]))
        return []


def create_custom_agent(env: EnvironmentBase) -> CustomAgent:
    return CustomAgent(env)
