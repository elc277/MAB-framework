# GENERAL NON-REPEATED SCALING EXPERIMENT
# Variable ratio: 1:r  =>  n agents, r*n arms
#Takes a long time to run !!!!

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
    """
    Builds a smooth non-repeated reward structure for any n_arms >= 1.
    The reward range widens gradually with scale.

    arms_per_agent is included so the function stays comparable across
    different ratio experiments if you want to extend it later.
    """
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
    max_n=30,
    arms_per_agent=2,
    steps=1000,
    save_dir=None,
    output_filename=None,
    base_seed=None,
    n_seeds=50,
    epsilons=None,
    std=1.0,
):
    """
    Runs scaling experiment for n = 1, 2, ..., max_n
    with configurations:
        n agents : arms_per_agent * n arms

    Parameters
    ----------
    max_n : int
        Maximum value of n in scaling family.
    arms_per_agent : int
        Number of arms per agent.
        Example:
            1 -> ratio 1:1
            2 -> ratio 1:2
            4 -> ratio 1:4
    steps : int
        Number of timesteps per run.
    save_dir : str | None
        Directory where results are saved.
    output_filename : str | None
        CSV filename.
    base_seed : int | None
        If provided, seeds will be deterministic:
            [base_seed + i for i in range(n_seeds)]
        If None, seeds will be:
            [0, 1, ..., n_seeds-1]
    n_seeds : int
        Number of seeds per condition.
    epsilons : list[float] | None
        Epsilon values to test.
    std : float
        Standard deviation for all arms.
    """
    if max_n < 1:
        raise ValueError("max_n must be at least 1")

    if arms_per_agent < 1:
        raise ValueError("arms_per_agent must be at least 1")

    if epsilons is None:
        epsilons = [
            0.00, 0.01, 0.02, 0.03, 0.04, 0.05,
            0.06, 0.07, 0.08, 0.09, 0.10,
            0.12, 0.14, 0.16, 0.18, 0.20,
            0.22, 0.24, 0.26, 0.28, 0.30,
            0.35, 0.40, 0.45, 0.50
        ]

    scaling_configs = [(n, arms_per_agent * n) for n in range(1, max_n + 1)]

    if base_seed is None:
        seeds = list(range(n_seeds))
    else:
        seeds = [base_seed + i for i in range(n_seeds)]

    if save_dir is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        save_dir = os.path.join(
            project_root,
            "results",
            f"experiment_ratio_1_to_{arms_per_agent}_1_to_{max_n}_agents"
        )

    if output_filename is None:
        output_filename = f"experiment_ratio_1_to_{arms_per_agent}_ratio_and_1_to_{max_n}_agents.csv"

    os.makedirs(save_dir, exist_ok=True)
    output_csv = os.path.join(save_dir, output_filename)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
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

        for n_agents, n_arms in scaling_configs:
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
                        f"Done | agents={n_agents} | arms={n_arms} | "
                        f"ratio=1:{arms_per_agent} | epsilon={epsilon:.2f} | "
                        f"seed={run_seed} | reward={group_total_reward:.2f} | "
                        f"collisions={collision_rate:.3f}"
                    )

    print(f"\nExperiment results saved to: {output_csv}")


if __name__ == "__main__":
    main()