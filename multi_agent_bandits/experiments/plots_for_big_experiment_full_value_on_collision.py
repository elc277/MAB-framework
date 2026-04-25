import os
import pandas as pd
import matplotlib.pyplot as plt


def main(
    csv_path,
    out_dir=None,
):
    if out_dir is None:
        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        out_dir = os.path.join("results", f"{csv_name}_plots")

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_columns = [
        "n_agents",
        "n_arms",
        "arms_per_agent",
        "epsilon",
        "group_total_reward",
        "mean_agent_reward",
        "collision_rate",
        "reward_inequality",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    policy_text = ""
    if "collision_policy" in df.columns:
        policies = sorted(df["collision_policy"].dropna().unique())
        if len(policies) == 1:
            policy_text = f" — {policies[0]}"

    unique_agents = sorted(df["n_agents"].dropna().unique())
    if len(unique_agents) == 1:
        agent_text = f" ({int(unique_agents[0])} agents, multiple ratios{policy_text})"
    else:
        agent_text = f" (multiple agent counts, multiple ratios{policy_text})"

    summary = (
        df.groupby(["n_agents", "n_arms", "arms_per_agent", "epsilon"], as_index=False)
        .agg(
            mean_group_total_reward=("group_total_reward", "mean"),
            std_group_total_reward=("group_total_reward", "std"),
            mean_mean_agent_reward=("mean_agent_reward", "mean"),
            std_mean_agent_reward=("mean_agent_reward", "std"),
            mean_collision_rate=("collision_rate", "mean"),
            std_collision_rate=("collision_rate", "std"),
            mean_reward_inequality=("reward_inequality", "mean"),
            std_reward_inequality=("reward_inequality", "std"),
        )
    )

    summary = summary.sort_values(["n_agents", "arms_per_agent", "n_arms", "epsilon"])

    summary["label"] = summary.apply(
        lambda row: f"{int(row['n_agents'])}A-{int(row['n_arms'])}K (1:{int(row['arms_per_agent'])})",
        axis=1
    )

    # 1. Group reward vs epsilon
    plt.figure(figsize=(10, 6))
    for label in summary["label"].unique():
        sub = summary[summary["label"] == label]
        plt.plot(sub["epsilon"], sub["mean_group_total_reward"], marker="o", label=label)
    plt.title(f"Average Group Total Reward vs Epsilon{agent_text}")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Group Total Reward")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "group_reward_vs_epsilon.png"), dpi=300)
    plt.show()

    # 2. Mean agent reward vs epsilon
    plt.figure(figsize=(10, 6))
    for label in summary["label"].unique():
        sub = summary[summary["label"] == label]
        plt.plot(sub["epsilon"], sub["mean_mean_agent_reward"], marker="o", label=label)
    plt.title(f"Average Reward Per Agent vs Epsilon{agent_text}")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Reward Per Agent")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_agent_reward_vs_epsilon.png"), dpi=300)
    plt.show()

    # 3. Collision rate vs epsilon
    plt.figure(figsize=(10, 6))
    for label in summary["label"].unique():
        sub = summary[summary["label"] == label]
        plt.plot(sub["epsilon"], sub["mean_collision_rate"], marker="o", label=label)
    plt.title(f"Average Collision Rate vs Epsilon{agent_text}")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Collision Rate")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "collision_rate_vs_epsilon.png"), dpi=300)
    plt.show()

    # 4. Reward inequality vs epsilon
    plt.figure(figsize=(10, 6))
    for label in summary["label"].unique():
        sub = summary[summary["label"] == label]
        plt.plot(sub["epsilon"], sub["mean_reward_inequality"], marker="o", label=label)
    plt.title(f"Average Reward Inequality vs Epsilon{agent_text}")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Reward Inequality")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_inequality_vs_epsilon.png"), dpi=300)
    plt.show()

    # 5. Normalized group reward vs epsilon
    norm_df = summary.copy()
    norm_df["normalized_group_reward"] = (
        norm_df.groupby("label")["mean_group_total_reward"]
        .transform(lambda x: x / x.max())
    )

    plt.figure(figsize=(10, 6))
    for label in norm_df["label"].unique():
        sub = norm_df[norm_df["label"] == label]
        plt.plot(sub["epsilon"], sub["normalized_group_reward"], marker="o", label=label)
    plt.title(f"Normalized Group Reward vs Epsilon{agent_text}")
    plt.xlabel("Epsilon")
    plt.ylabel("Normalized Group Reward")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "normalized_group_reward_vs_epsilon.png"), dpi=300)
    plt.show()

    # Best epsilon table
    best = summary.loc[summary.groupby("label")["mean_group_total_reward"].idxmax()].copy()
    best = best.sort_values(["arms_per_agent", "n_agents", "n_arms"])

    # 6. Best epsilon by configuration
    plt.figure(figsize=(11, 5))
    plt.plot(best["label"], best["epsilon"], marker="o")
    plt.title(f"Best Epsilon by Configuration{agent_text}")
    plt.xlabel("Configuration")
    plt.ylabel("Best Epsilon")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_epsilon_by_configuration.png"), dpi=300)
    plt.show()

    # 7. Best epsilon vs ratio
    plt.figure(figsize=(8, 5))
    plt.plot(best["arms_per_agent"], best["epsilon"], marker="o")
    plt.title(f"Best Epsilon vs Ratio{agent_text}")
    plt.xlabel("Arms per Agent (ratio 1:x)")
    plt.ylabel("Best Epsilon")
    plt.xticks(best["arms_per_agent"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_epsilon_vs_ratio.png"), dpi=300)
    plt.show()

    print(f"Plots saved in: {out_dir}")


if __name__ == "__main__":
    main(
        csv_path=r"results/big_experiment_100_agents_full_value_collision/big_experiment_100_agents_full_value_collision.csv",
        out_dir=r"results/big_experiment_100_agents_full_value_collision/plots"
    )