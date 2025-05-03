# vi_agent.py

from mdp import MDPRescueSimple
from rl_utils import BaseAgent
import models as mdl

class ValueIterationAgent(BaseAgent):
    """
    In-place (Gauss–Seidel) Value Iteration for RescueSimple.
    """

    DISCOUNT = 0.9

    def __init__(self, mdp: MDPRescueSimple):
        super().__init__()
        self.mdp = mdp
        # initialize V(s)=0
        self.V = {s: 0.0 for s in self.mdp.all_states()}

        theta = 1e-3
        # In-place update until convergence
        while True:
            delta = 0.0
            for s in self.mdp.all_states():
                if self.mdp.is_terminal(s):
                    continue
                # compute max_a Σ P [r + γ V[s′]]
                best_q = float('-inf')
                for a in self.mdp.possible_actions(s):
                    q = 0.0
                    for prob, s2 in self.mdp.get_transition_distribution(s, a):
                        r = self.mdp.get_reward(s, a, s2)
                        q += prob * (r + self.DISCOUNT * self.V[s2])
                    if q > best_q:
                        best_q = q
                delta = max(delta, abs(self.V[s] - best_q))
                self.V[s] = best_q
            if delta < theta:
                break

    def get_Vvalue(self, state: mdl.StateRescue):
        return self.V.get(state, 0.0)

    def get_Qvalue(self, state: mdl.StateRescue, action):
        q = 0.0
        for prob, s2 in self.mdp.get_transition_distribution(state, action):
            r = self.mdp.get_reward(state, action, s2)
            q += prob * (r + self.DISCOUNT * self.V[s2])
        return q

    def get_best_action(self, state: mdl.StateRescue):
        if self.mdp.is_terminal(state):
            return mdl.Action.STAY
        best_a, best_q = mdl.Action.STAY, float('-inf')
        for a in self.mdp.possible_actions(state):
            q = self.get_Qvalue(state, a)
            if q > best_q:
                best_q, best_a = q, a
        return best_a