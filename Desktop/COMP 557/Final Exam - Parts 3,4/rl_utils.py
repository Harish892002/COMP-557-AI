from typing import Callable
import models as mdl
from env import EnvironmentBase
from tqdm import tqdm


class BaseAgent:

    def get_best_action(self, state: mdl.StateRescue):
        '''
        Get the best action for a given state from the learned policy.
        '''
        raise NotImplementedError("Method not implemented.")


class RLAgent(BaseAgent):

    def get_best_action(self, state: mdl.StateRescue):
        '''
        Get the best action for a given state from the learned policy.
        '''
        return self.get_action(state, deterministic=True)

    def update(self, state: mdl.StateRescue, action, reward,
               next_state: mdl.StateRescue):
        """
        Update Q-values using the Q-learning formula
        """
        raise NotImplementedError("Method not implemented.")

    def get_action(self, state: mdl.StateRescue, deterministic=False):
        '''
        deterministic: If True, choose the best action.
                       If False, choose an action for RL training
                                    (e.g., epsilon-greedy policy).
        '''
        raise NotImplementedError("Method not implemented.")

    def get_num_episodes_to_train(self) -> int:
        """
        Return the number of episodes to train the agent.
        """
        raise NotImplementedError("Method not implemented.")


def train_rl_agent(env: EnvironmentBase, agent: RLAgent):
    """
    Training loop for the RL agent.
    """
    num_episodes = agent.get_num_episodes_to_train()
    for idx in tqdm(range(num_episodes)):
        state = env.reset()
        while not env.terminated():
            action = agent.get_action(state)
            next_state, reward = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state


def eval_agent(env: EnvironmentBase, agent: BaseAgent, n_episodes=10):
    """
    Implement the evaluation loop for the RL agent.
    """
    returns = []
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        while not env.terminated():
            action = agent.get_best_action(state)
            next_state, reward = env.step(action)
            total_reward += reward
            state = next_state
        returns.append(total_reward)

    return returns
