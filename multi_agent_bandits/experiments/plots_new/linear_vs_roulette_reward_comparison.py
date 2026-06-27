import os
import pandas as pd
import matplotlib.pyplot as plt


def get_best_reward_by_ratio(csv_path, policy_label):
    df = pd.read_csv(csv_path)

    df_100 = df[df["n_agents"] == 100].copy()

    summary = (
        df_100.groupby(["target_ratio", "epsilon"], as_index=False)
        .agg(
            mean_group_total_reward=("group_total_reward", "mean"),
            mean_mean_agent_reward=("mean_agent_reward", "mean"),
            mean_collision_rate=("collision_rate", "mean"),
        )
    )

    best = summary.loc[
        summary.groupby("target_ratio")["mean_group_total_reward"].idxmax()
    ].copy()

    best["policy"] = policy_label

    return best


def ratio_sort_key(ratio):
    left, right = ratio.split(":")
    return float(left) / float(right)


def main():
    linear_csv = r"results/full_ratio_sweep_1_to_100_agents_linear_collision_multicore/full_ratio_sweep_1_to_100_agents_linear_collision_multicore.csv"
    roulette_csv = r"results/full_ratio_sweep_1_to_100_agents_random_roulette_multicore/full_ratio_sweep_1_to_100_agents_random_roulette_collision_multicore.csv"

    out_dir = r"results/linear_vs_roulette_reward_comparison"
    os.makedirs(out_dir, exist_ok=True)

    linear_best = get_best_reward_by_ratio(linear_csv, "linear_share")
    roulette_best = get_best_reward_by_ratio(roulette_csv, "random_roulette")

    combined = pd.concat([linear_best, roulette_best], ignore_index=True)

    combined["ratio_sort"] = combined["target_ratio"].apply(ratio_sort_key)
    combined = combined.sort_values(["ratio_sort", "policy"])

    combined.to_csv(
        os.path.join(out_dir, "linear_vs_roulette_best_reward_by_ratio_100_agents.csv"),
        index=False
    )

    # Plot group reward at each policy's best epsilon
    plt.figure(figsize=(9, 5))

    for policy in ["linear_share", "random_roulette"]:
        sub = combined[combined["policy"] == policy].sort_values("ratio_sort")
        plt.plot(
            sub["target_ratio"],
            sub["mean_group_total_reward"],
            marker="o",
            label=policy
        )

    plt.title("Linear Share vs Random Roulette: Group Reward at Best Epsilon")
    plt.xlabel("Target Ratio (Agents:Arms)")
    plt.ylabel("Average Group Total Reward")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "linear_vs_roulette_group_reward_at_best_epsilon.png"),
        dpi=300
    )
    plt.show()

    # Plot best epsilon too, for direct comparison
    plt.figure(figsize=(9, 5))

    for policy in ["linear_share", "random_roulette"]:
        sub = combined[combined["policy"] == policy].sort_values("ratio_sort")
        plt.plot(
            sub["target_ratio"],
            sub["epsilon"],
            marker="o",
            label=policy
        )

    plt.title("Linear Share vs Random Roulette: Best Epsilon by Ratio")
    plt.xlabel("Target Ratio (Agents:Arms)")
    plt.ylabel("Best Epsilon")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "linear_vs_roulette_best_epsilon_by_ratio_100_agents.png"),
        dpi=300
    )
    plt.show()

    print(f"Saved comparison outputs in: {out_dir}")
    print(combined[[
        "policy",
        "target_ratio",
        "epsilon",
        "mean_group_total_reward",
        "mean_mean_agent_reward",
        "mean_collision_rate"
    ]])


if __name__ == "__main__":
    main()