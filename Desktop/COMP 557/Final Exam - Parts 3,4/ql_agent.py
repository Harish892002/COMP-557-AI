# ql_agent.py

from typing import Callable
import models as mdl
from rl_utils import RLAgent
import random

class QLearningAgent(RLAgent):
    DISCOUNT = 0.9

    def __init__(
        self,
        cb_all_states: Callable[[], list],
        cb_possible_actions: Callable[[mdl.StateRescue], list],
        alpha: float = 0.1,
        epsilon: float = 0.1,
        n_episodes: int = 200,
        planning_steps: int = 2,
    ):
        super().__init__()
        self.states = cb_all_states()
        self.actions = cb_possible_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.planning_steps = planning_steps

        # Q-table
        self.Q = {
            s: {a: 0.0 for a in self.actions(s)}
            for s in self.states
        }

        # simple model for Dyna-Q
        self.model = {}

    def get_Qvalue(self, state, action):
        return self.Q[state][action]

    def update(self, state, action, reward, next_state):
        # 1) real experience
        best_next = 0.0
        nxt = self.actions(next_state)
        if nxt:
            best_next = max(self.Q[next_state][a2] for a2 in nxt)
        old = self.Q[state][action]
        self.Q[state][action] = old + self.alpha * (
            reward + self.DISCOUNT * best_next - old
        )

        # 2) store in model
        self.model[(state, action)] = (reward, next_state)

        # 3) a couple of planning steps
        for _ in range(self.planning_steps):
            (s0, a0), (r0, s0p) = random.choice(list(self.model.items()))
            best = 0.0
            nas = self.actions(s0p)
            if nas:
                best = max(self.Q[s0p][a2] for a2 in nas)
            old2 = self.Q[s0][a0]
            self.Q[s0][a0] = old2 + self.alpha * (
                r0 + self.DISCOUNT * best - old2
            )

    def get_action(self, state, deterministic=False):
        acts = self.actions(state)
        if not acts:
            return mdl.Action.STAY
        if deterministic or random.random() > self.epsilon:
            return max(acts, key=lambda a: self.Q[state][a])
        return random.choice(acts)

    def get_num_episodes_to_train(self):
        return self.n_episodes