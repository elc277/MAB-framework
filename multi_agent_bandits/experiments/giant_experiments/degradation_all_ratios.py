# FULL RATIO SWEEP EXPERIMENT - DEGRADATION COLLISION
# MULTICORE VERSION: uses 6 CPU cores by default
# Collision policy: equal split of a degraded reward
# Ratios tested:
# 20:1, 5:1, 4:1, 3:1, 2:1, 1:1, 1:2, 1:3, 1:4, 1:5, 1:20
# For each ratio: n_agents = 1 to 100
# Epsilons tested across 50 seeds

import os
import sys
import csv
import random
import statistics
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from multi_agent_bandits.core.environment import Environment
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.arm import Arm
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent


def degradation_collision(raw_reward, n_agents, degradation_strength=0.5):
    """
    Equal split of a degraded reward.

    Formula:
        degraded_total_reward = raw_reward / (1 + degradation_strength * (n_agents - 1))
        each_agent_reward = degraded_total_reward / n_agents
    """
    degraded_total_reward = raw_reward / (1 + degradation_strength * (n_agents - 1))
    share = degraded_total_reward / n_agents
    return [share] * n_agents


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


def build_arms_for_config(n_arms, std):
    """
    Smooth non-repeated reward structure for any number of arms.
    """
    if n_arms < 1:
        raise ValueError("n_arms must be at least 1")

    if n_arms == 1:
        means = [1.0]
    else:
        scale_index = max(0, n_arms - 1)

        lower = max(0.1, 1.0 - 0.01 * scale_index)
        upper = 2.0 + 0.01 * scale_index

        means = np.linspace(lower, upper, n_arms).tolist()

    return [Arm(mean=m, sd=std) for m in means]


def compute_n_arms(n_agents, agent_ratio_part, arm_ratio_part):
    return max(1, int(round(n_agents * arm_ratio_part / agent_ratio_part)))


def run_single_experiment(task):
    (
        agent_ratio_part,
        arm_ratio_part,
        target_ratio_label,
        n_agents,
        n_arms,
        actual_ratio,
        epsilon,
        run_seed,
        steps,
        std,
        degradation_strength,
    ) = task

    random.seed(run_seed)
    np.random.seed(run_seed)

    arms = build_arms_for_config(
        n_arms=n_arms,
        std=std
    )

    def collision_policy(raw_reward, n_colliding_agents):
        return degradation_collision(
            raw_reward,
            n_colliding_agents,
            degradation_strength=degradation_strength
        )

    env = Environment(
        n_agents=n_agents,
        arms=arms,
        collision_policy=collision_policy
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

    with redirect_stdout(io.StringIO()):
        choices_log, rewards_log = runner.run(
            plot_rewards=False,
            plot_frequencies=False
        )

    group_total_reward = sum(runner.total_rewards)
    mean_agent_reward = group_total_reward / n_agents
    collision_rate = compute_collision_rate(choices_log)
    reward_inequality = compute_reward_inequality(runner.total_rewards)

    return [
        "degradation",
        degradation_strength,
        target_ratio_label,
        agent_ratio_part,
        arm_ratio_part,
        n_agents,
        n_arms,
        actual_ratio,
        std,
        epsilon,
        run_seed,
        steps,
        group_total_reward,
        mean_agent_reward,
        collision_rate,
        reward_inequality
    ]


def main(
    max_agents=100,
    ratio_pairs=None,
    steps=1000,
    save_dir=None,
    output_filename=None,
    base_seed=None,
    n_seeds=50,
    epsilons=None,
    std=1.0,
    degradation_strength=0.5,
    max_workers=6,
):

    if max_agents < 1:
        raise ValueError("max_agents must be at least 1")

    if ratio_pairs is None:
        ratio_pairs = [
            (20, 1),
            (5, 1),
            (4, 1),
            (3, 1),
            (2, 1),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 20),
        ]

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
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        save_dir = os.path.join(
            project_root,
            "results",
            f"full_ratio_sweep_1_to_{max_agents}_agents_degradation_collision_multicore"
        )

    if output_filename is None:
        output_filename = f"full_ratio_sweep_1_to_{max_agents}_agents_degradation_collision_multicore.csv"

    os.makedirs(save_dir, exist_ok=True)
    output_csv = os.path.join(save_dir, output_filename)

    tasks = []

    for agent_ratio_part, arm_ratio_part in ratio_pairs:
        target_ratio_label = f"{agent_ratio_part}:{arm_ratio_part}"

        for n_agents in range(1, max_agents + 1):
            n_arms = compute_n_arms(
                n_agents=n_agents,
                agent_ratio_part=agent_ratio_part,
                arm_ratio_part=arm_ratio_part
            )

            actual_ratio = n_agents / n_arms

            for epsilon in epsilons:
                for run_seed in seeds:
                    tasks.append((
                        agent_ratio_part,
                        arm_ratio_part,
                        target_ratio_label,
                        n_agents,
                        n_arms,
                        actual_ratio,
                        epsilon,
                        run_seed,
                        steps,
                        std,
                        degradation_strength,
                    ))

    total_tasks = len(tasks)

    print(f"CSV will be saved to: {output_csv}")
    print(f"Total runs: {total_tasks}")
    print(f"Using {max_workers} worker processes")
    print(f"Degradation strength: {degradation_strength}")
    print("Progress will print every 1000 completed runs.")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "collision_policy",
            "degradation_strength",
            "target_ratio",
            "target_agent_part",
            "target_arm_part",
            "n_agents",
            "n_arms",
            "actual_agent_to_arm_ratio",
            "std",
            "epsilon",
            "seed",
            "steps",
            "group_total_reward",
            "mean_agent_reward",
            "collision_rate",
            "reward_inequality"
        ])
        f.flush()

        completed = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_single_experiment, task) for task in tasks]

            for future in as_completed(futures):
                row = future.result()
                writer.writerow(row)
                f.flush()

                completed += 1

                if completed % 1000 == 0 or completed == total_tasks:
                    print(f"[{completed}/{total_tasks}] completed")

    print(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main(max_workers=6)