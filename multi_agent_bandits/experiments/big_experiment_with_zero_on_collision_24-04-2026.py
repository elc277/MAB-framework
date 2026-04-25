# FIXED N EXPERIMENT - COLLISION POLICY 2
# Nobody gets anything on collision / zero-on-collision
# n = 100 agents
# Ratios tested: 1:1, 1:2, 1:3, 1:4, 1:5, 1:20
# Collision policy 2:
# Nobody gets reward on collision

import os
import sys
import csv
import random
import statistics
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from multi_agent_bandits.core.environment import Environment
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.arm import Arm
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent


def zero_on_collision(raw_reward, n_agents):
    return [0.0] * n_agents


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


def build_arms_for_config(n_arms, std, arms_per_agent):
    if n_arms < 1:
        raise ValueError("n_arms must be at least 1")

    if n_arms == 1:
        means = [1.0]
    else:
        scale_index = max(0, n_arms // max(1, arms_per_agent) - 1)

        lower = max(0.1, 1.0 - 0.1 * scale_index)
        upper = 2.0 + 0.1 * scale_index

        means = np.linspace(lower, upper, n_arms).tolist()

    return [Arm(mean=m, sd=std) for m in means]


def main(
    n_agents=100,
    ratios=(1, 2, 3, 4, 5, 20),
    steps=1000,
    save_dir=None,
    output_filename=None,
    base_seed=None,
    n_seeds=50,
    epsilons=None,
    std=1.0,
):

    if n_agents < 1:
        raise ValueError("n_agents must be at least 1")

    if epsilons is None:
        epsilons = [
            0.01, 0.02, 0.03, 0.04, 0.05,
            0.06, 0.07, 0.08, 0.09, 0.10,
            0.12, 0.14, 0.16, 0.18, 0.20,
            0.22, 0.24, 0.26, 0.28, 0.30,
            0.35, 0.40, 0.45, 0.50
        ]

    if base_seed is None:
        seeds = list(range(n_seeds))
    else:
        seeds = [base_seed + i for i in range(n_seeds)]

    if save_dir is None:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        save_dir = os.path.join(
            project_root,
            "results",
            f"big_experiment_{n_agents}_agents_zero_on_collision"
        )

    if output_filename is None:
        output_filename = f"big_experiment_{n_agents}_agents_zero_on_collision.csv"

    os.makedirs(save_dir, exist_ok=True)
    output_csv = os.path.join(save_dir, output_filename)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "collision_policy",
            "n_agents",
            "n_arms",
            "agent_to_arm_ratio",
            "arms_per_agent",
            "std",
            "epsilon",
            "seed",
            "steps",
            "group_total_reward",
            "mean_agent_reward",
            "collision_rate",
            "reward_inequality"
        ])

        for arms_per_agent in ratios:
            n_arms = n_agents * arms_per_agent
            ratio = n_agents / n_arms

            for epsilon in epsilons:
                for run_seed in seeds:
                    random.seed(run_seed)
                    np.random.seed(run_seed)

                    arms = build_arms_for_config(
                        n_arms=n_arms,
                        std=std,
                        arms_per_agent=arms_per_agent
                    )

                    env = Environment(
                        n_agents=n_agents,
                        arms=arms,
                        collision_policy=zero_on_collision
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
                    reward_inequality = compute_reward_inequality(
                        runner.total_rewards
                    )

                    writer.writerow([
                        "zero_on_collision",
                        n_agents,
                        n_arms,
                        ratio,
                        arms_per_agent,
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
                        f"Done | policy=zero_on_collision | agents={n_agents} | arms={n_arms} | "
                        f"ratio=1:{arms_per_agent} | epsilon={epsilon:.2f} | "
                        f"seed={run_seed} | reward={group_total_reward:.2f} | "
                        f"collisions={collision_rate:.3f}"
                    )

    print(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    main()