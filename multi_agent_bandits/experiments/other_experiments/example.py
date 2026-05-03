from multi_agent_bandits.core.environment import Environment
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.arm import Arm

from multi_agent_bandits.strategies.ucb_baseline import UCB_BaselineAgent
from multi_agent_bandits.strategies.random import RandomAgent
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent

import random
import numpy as np

def main(steps=1000, save_dir=None, plot_rewards=False, plot_frequencies=False, seed=None):
    if seed is not None:
        random.seed(seed)
        try:
            np.random.seed(seed)
        except Exception:
            pass
    n_agents = 3

    arms = [
        Arm(mean=1.0, sd=1.0),
        Arm(mean=2.0, sd=1.0),
        Arm(mean=1.5, sd=1.0)
    ]

    env = Environment(
        n_agents=n_agents,
        arms=arms
    )

    agents = [
        RandomAgent(env.n_arms),
        EpsilonGreedyAgent(env.n_arms),
        UCB_BaselineAgent(env.n_arms)
    ]

    runner = ExperimentRunner(
        env,
        agents,
        timestep_limit=steps,
        save_dir=save_dir
    )

    runner.run(
        plot_rewards=plot_rewards,
        plot_frequencies=plot_frequencies
    )

    runner.print_summary()
