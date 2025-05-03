import argparse
from vi_agent import ValueIterationAgent
from ql_agent import QLearningAgent
from custom_agent import create_custom_agent
from env import EnvRescueComplex, EnvRescueSimple, EnvironmentBase
from app import GridWorldApp
import random
from models import MAP_V1
from rl_utils import eval_agent, train_rl_agent
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

ENVS = {"simple": EnvRescueSimple, "complex": EnvRescueComplex}


def random_boundary_coord(h, w):
    boundary_coords = []

    # Top and bottom rows
    for col in range(w):
        boundary_coords.append((0, col))  # Top row
        boundary_coords.append((h - 1, col))  # Bottom row

    # Left and right columns (excluding corners to avoid duplicates)
    for row in range(1, h - 1):
        boundary_coords.append((row, 0))  # Left column
        boundary_coords.append((row, w - 1))  # Right column

    return random.choice(boundary_coords)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL agent")
    parser.add_argument("--agent",
                        type=str,
                        choices=["vi", "ql", "ca", "manual"],
                        default="vi",
                        help="Agent type")
    parser.add_argument("--env",
                        type=str,
                        choices=ENVS.keys(),
                        required=True,
                        help="Environment type")
    parser.add_argument("--gui",
                        action="store_true",
                        help="Run with GUI visualization")
    parser.add_argument("--n_episodes",
                        type=int,
                        default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--n_objs",
                        type=int,
                        default=3,
                        help="Number of the objects for the complex domain")
    args = parser.parse_args()

    env_class = ENVS[args.env]
    obj_loc = random_boundary_coord(MAP_V1['height'], MAP_V1['width'])
    env_config = dict(obj_loc=obj_loc, n_objs=args.n_objs, **MAP_V1)
    env = env_class(**env_config)  # type: EnvironmentBase

    if args.agent == "vi":
        agent = ValueIterationAgent(env.mdp)
    elif args.agent == "ql":
        agent = QLearningAgent(env.mdp.all_states, env.mdp.possible_actions)
        train_rl_agent(env, agent)
    elif args.agent == "ca":
        agent = create_custom_agent(env)
    elif args.agent == "manual":
        agent = None
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")

    if args.gui:
        app = GridWorldApp(env, agent)
        app.run()
    else:
        if agent is not None:
            rewards = eval_agent(env, agent, n_episodes=args.n_episodes)
            print(rewards)
        else:
            print("No GUI mode selected. Exiting.")
