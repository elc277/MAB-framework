import os
import sys

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


def build_arms(n_arms, std):
    base_means = [1.0, 1.5, 2.0]
    means = [base_means[i % len(base_means)] for i in range(n_arms)]
    return [Arm(mean=m, sd=std) for m in means]


def main(steps=1000, save_dir=None, seed=None):
    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    std = 1.0
    scaling_configs = [
        (1, 3), #1 agent, 3 arms
        (2, 6), #2 agents, 6 arms
        (3, 9), #3 agents, 9 arms
    ]
    seeds = list(range(50)) if seed is None else [seed]

    if save_dir is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        save_dir = os.path.join(project_root, "results", "pilot_experiment_9-04-2026")

    os.makedirs(save_dir, exist_ok=True)
    output_csv = os.path.join(save_dir, "scaling_summary.csv")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n_agents",
            "n_arms",
            "agent_to_arm_ratio",
            "std",
            "epsilon",
            "seed",
            "steps",
            "group_total_reward",
            "mean_agent_reward",
            "collision_rate",
            "reward_inequality"
        ])

        for n_agents, n_arms in scaling_configs:
            ratio = n_agents / n_arms

            for epsilon in epsilons:
                for run_seed in seeds:
                    random.seed(run_seed)
                    np.random.seed(run_seed)

                    arms = build_arms(n_arms, std)

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
                        n_arms,
                        ratio,
                        std,
                        epsilon,
                        run_seed,
                        steps,
                        group_total_reward,
                        mean_agent_reward,
                        collision_rate,
                        reward_inequality
                    ])

                    print(
                        f"Done | agents={n_agents} | arms={n_arms} | epsilon={epsilon:.2f} | "
                        f"seed={run_seed} | reward={group_total_reward:.2f} | "
                        f"collisions={collision_rate:.3f}"
                    )

    print(f"\nScaling experiment finished. Results saved to: {output_csv}")


if __name__ == "__main__":
    main()