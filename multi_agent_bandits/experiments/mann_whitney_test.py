import os
import pandas as pd
from scipy.stats import mannwhitneyu


# ============================================================
# File paths
# Replace these paths with your actual CSV paths
# ============================================================

CSV_PATHS = {
    "full_reward": r"results/full_ratio_sweep_1_to_100_agents_full_value_collision_multicore/full_ratio_sweep_1_to_100_agents_full_value_collision_multicore.csv",
    "linear_share": r"results/full_ratio_sweep_1_to_100_agents_linear_collision_multicore/full_ratio_sweep_1_to_100_agents_linear_collision_multicore.csv",
    "random_roulette": r"results/full_ratio_sweep_1_to_100_agents_random_roulette_multicore/full_ratio_sweep_1_to_100_agents_random_roulette_collision_multicore.csv",
    "zero_on_collision": r"results/full_ratio_sweep_1_to_100_agents_zero_on_collision_multicore/full_ratio_sweep_1_to_100_agents_zero_on_collision_multicore.csv",
    "degradation": r"results/full_ratio_sweep_1_to_100_agents_degradation_collision_multicore_part_1_under/full_ratio_sweep_degradation_complete.csv",
}

OUT_DIR = r"results/statistical_tests"
OUT_FILE = "mann_whitney_results.csv"

RATIO_ORDER = [
    "20:1", "5:1", "4:1", "3:1", "2:1",
    "1:1",
    "1:2", "1:3", "1:4", "1:5", "1:20"
]

METRIC = "group_total_reward"
N_AGENTS = 100


# ============================================================
# Helper functions
# ============================================================

def load_csv(policy_name):
    path = CSV_PATHS[policy_name]

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found for {policy_name}: {path}")

    df = pd.read_csv(path)

    required_columns = [
        "collision_policy",
        "target_ratio",
        "n_agents",
        "n_arms",
        "epsilon",
        "seed",
        "group_total_reward",
        "mean_agent_reward",
        "collision_rate",
        "reward_inequality",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {policy_name}: {missing}")

    return df


def get_best_epsilon(df, ratio, n_agents=N_AGENTS, metric=METRIC):
    """
    Finds the epsilon with the highest mean metric across seeds
    for one ratio and one number of agents.
    """
    sub = df[
        (df["target_ratio"] == ratio) &
        (df["n_agents"] == n_agents)
    ].copy()

    if sub.empty:
        raise ValueError(f"No data found for ratio={ratio}, n_agents={n_agents}")

    summary = (
        sub.groupby("epsilon", as_index=False)
        .agg(mean_metric=(metric, "mean"))
        .sort_values("mean_metric", ascending=False)
    )

    return float(summary.iloc[0]["epsilon"])


def get_seed_values(df, ratio, epsilon, n_agents=N_AGENTS, metric=METRIC):
    """
    Returns the seed-level values for a specific ratio, epsilon, and number of agents.
    """
    sub = df[
        (df["target_ratio"] == ratio) &
        (df["n_agents"] == n_agents) &
        (df["epsilon"] == epsilon)
    ].copy()

    if sub.empty:
        raise ValueError(
            f"No data found for ratio={ratio}, epsilon={epsilon}, n_agents={n_agents}"
        )

    return sub[metric].dropna()


def mann_whitney_test(values_a, values_b, alternative="two-sided"):
    """
    Runs a Mann-Whitney U test.
    alternative can be:
    - 'two-sided'
    - 'greater'
    - 'less'
    """
    result = mannwhitneyu(values_a, values_b, alternative=alternative)

    return result.statistic, result.pvalue


def add_result(
    rows,
    comparison_type,
    comparison,
    ratio,
    policy_a,
    epsilon_a,
    policy_b,
    epsilon_b,
    values_a,
    values_b,
    alternative,
    u_stat,
    p_value,
):
    rows.append({
        "comparison_type": comparison_type,
        "comparison": comparison,
        "ratio": ratio,
        "n_agents": N_AGENTS,
        "metric": METRIC,
        "policy_a": policy_a,
        "epsilon_a": epsilon_a,
        "n_a": len(values_a),
        "mean_a": values_a.mean(),
        "median_a": values_a.median(),
        "policy_b": policy_b,
        "epsilon_b": epsilon_b,
        "n_b": len(values_b),
        "mean_b": values_b.mean(),
        "median_b": values_b.median(),
        "alternative": alternative,
        "u_statistic": u_stat,
        "p_value": p_value,
    })


# ============================================================
# Test set 1:
# Linear share vs random roulette at 100 agents, all ratios
# Each policy is tested at its own empirically optimal epsilon
# ============================================================

def test_linear_vs_random_roulette(rows):
    linear_df = load_csv("linear_share")
    roulette_df = load_csv("random_roulette")

    for ratio in RATIO_ORDER:
        linear_best_eps = get_best_epsilon(linear_df, ratio)
        roulette_best_eps = get_best_epsilon(roulette_df, ratio)

        linear_values = get_seed_values(linear_df, ratio, linear_best_eps)
        roulette_values = get_seed_values(roulette_df, ratio, roulette_best_eps)

        u_stat, p_value = mann_whitney_test(
            linear_values,
            roulette_values,
            alternative="two-sided"
        )

        add_result(
            rows=rows,
            comparison_type="policy_comparison",
            comparison="linear_share_vs_random_roulette_at_best_epsilon",
            ratio=ratio,
            policy_a="linear_share",
            epsilon_a=linear_best_eps,
            policy_b="random_roulette",
            epsilon_b=roulette_best_eps,
            values_a=linear_values,
            values_b=roulette_values,
            alternative="two-sided",
            u_stat=u_stat,
            p_value=p_value,
        )


# ============================================================
# Test set 2:
# Degradation high/best epsilon vs low epsilon
# For moderate crowded ratios: 5:1, 4:1, 3:1
# This tests whether high exploration improves reward there.
# ============================================================

def test_degradation_best_vs_low_epsilon(rows):
    degradation_df = load_csv("degradation")

    moderate_ratios = ["5:1", "4:1", "3:1"]
    low_epsilon = 0.01

    for ratio in moderate_ratios:
        best_eps = get_best_epsilon(degradation_df, ratio)

        best_values = get_seed_values(degradation_df, ratio, best_eps)
        low_values = get_seed_values(degradation_df, ratio, low_epsilon)

        # Alternative='greater' tests whether best/high epsilon has larger reward than low epsilon.
        u_stat, p_value = mann_whitney_test(
            best_values,
            low_values,
            alternative="greater"
        )

        add_result(
            rows=rows,
            comparison_type="within_policy_epsilon_comparison",
            comparison="degradation_best_epsilon_vs_low_epsilon",
            ratio=ratio,
            policy_a="degradation",
            epsilon_a=best_eps,
            policy_b="degradation",
            epsilon_b=low_epsilon,
            values_a=best_values,
            values_b=low_values,
            alternative="greater",
            u_stat=u_stat,
            p_value=p_value,
        )


# ============================================================
# Test set 3:
# Zero-on-collision low epsilon vs higher epsilon
# For crowded ratios: 20:1, 5:1, 4:1, 3:1, 2:1
# This tests whether increasing epsilon reduces reward.
# ============================================================

def test_zero_on_collision_low_vs_higher_epsilon(rows):
    zero_df = load_csv("zero_on_collision")

    crowded_ratios = ["20:1", "5:1", "4:1", "3:1", "2:1"]

    low_epsilon = 0.01
    higher_epsilon = 0.10

    for ratio in crowded_ratios:
        low_values = get_seed_values(zero_df, ratio, low_epsilon)
        higher_values = get_seed_values(zero_df, ratio, higher_epsilon)

        # Alternative='greater' tests whether low epsilon has larger reward than higher epsilon.
        u_stat, p_value = mann_whitney_test(
            low_values,
            higher_values,
            alternative="greater"
        )

        add_result(
            rows=rows,
            comparison_type="within_policy_epsilon_comparison",
            comparison="zero_on_collision_low_epsilon_vs_higher_epsilon",
            ratio=ratio,
            policy_a="zero_on_collision",
            epsilon_a=low_epsilon,
            policy_b="zero_on_collision",
            epsilon_b=higher_epsilon,
            values_a=low_values,
            values_b=higher_values,
            alternative="greater",
            u_stat=u_stat,
            p_value=p_value,
        )


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []

    test_linear_vs_random_roulette(rows)
    test_degradation_best_vs_low_epsilon(rows)
    test_zero_on_collision_low_vs_higher_epsilon(rows)

    results = pd.DataFrame(rows)

    # Optional: add a simple significance label
    results["significant_p_0_05"] = results["p_value"] < 0.05
    results["significant_p_0_01"] = results["p_value"] < 0.01
    results["significant_p_0_001"] = results["p_value"] < 0.001

    output_path = os.path.join(OUT_DIR, OUT_FILE)
    results.to_csv(output_path, index=False)

    print(f"Saved Mann-Whitney test results to: {output_path}")
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()