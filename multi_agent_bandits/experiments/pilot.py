import os
import sys

# Make the project root importable when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from multi_agent_bandits.core.environment import Environment
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.arm import Arm
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent

import random
import numpy as np
import csv
import statistics


def compute_collision_rate(choices_log):
    collision_steps = 0
    for step in choices_log:
        if len(set(step)) < len(step):
            collision_steps += 1
    return collision_steps / len(choices_log) if choices_log else 0.0


def compute_reward_inequality(total_rewards):
    if len(total_rewards) <= 1:
        return 0.0
    return statistics.pstdev(total_rewards)


def main(steps=1000, save_dir=None, seed=None):
    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    agent_counts = [2, 3, 4, 6]
    seeds = list(range(50)) if seed is None else [seed]

    if save_dir is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        save_dir = os.path.join(project_root, "results", "pilot")

    os.makedirs(save_dir, exist_ok=True)
    output_csv = os.path.join(save_dir, "pilot_summary.csv")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n_agents",
            "n_arms",
            "epsilon",
            "seed",
            "steps",
            "group_total_reward",
            "mean_agent_reward",
            "collision_rate",
            "reward_inequality"
        ])

        for n_agents in agent_counts:
            for epsilon in epsilons:
                for run_seed in seeds:
                    random.seed(run_seed)
                    np.random.seed(run_seed)

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
                        EpsilonGreedyAgent(
                            env.n_arms,
                            epsilon=epsilon,
                            name=f"EGreedy(eps={epsilon})"
                        )
                        for _ in range(n_agents)
                    ]

                    runner = ExperimentRunner(
                        env,
                        agents,
                        timestep_limit=steps,
                        save_dir=None
                    )

                    choices_log, rewards_log = runner.run(
                        plot_rewards=False,
                        plot_frequencies=False
                    )

                    group_total_reward = sum(runner.total_rewards)
                    mean_agent_reward = group_total_reward / n_agents
                    collision_rate = compute_collision_rate(choices_log)
                    reward_inequality = compute_reward_inequality(runner.total_rewards)

                    writer.writerow([
                        n_agents,
                        env.n_arms,
                        epsilon,
                        run_seed,
                        steps,
                        group_total_reward,
                        mean_agent_reward,
                        collision_rate,
                        reward_inequality
                    ])

                    print(
                        f"Done | agents={n_agents} | epsilon={epsilon:.2f} | seed={run_seed} "
                        f"| group_reward={group_total_reward:.2f} | collisions={collision_rate:.3f}"
                    )

    print(f"\nPilot finished. Results saved to: {output_csv}")


if __name__ == "__main__":
    main()