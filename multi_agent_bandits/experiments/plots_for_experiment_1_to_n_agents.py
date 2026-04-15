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
        "epsilon",
        "group_total_reward",
        "mean_agent_reward",
        "collision_rate",
        "reward_inequality",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    summary = (
        df.groupby(["n_agents", "n_arms", "epsilon"], as_index=False)
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

    summary = summary.sort_values(["n_agents", "n_arms", "epsilon"])
    summary["label"] = summary.apply(
        lambda row: f"{int(row['n_agents'])}A-{int(row['n_arms'])}K",
        axis=1
    )

    # 1. Group reward vs epsilon
    plt.figure(figsize=(10, 6))
    for label in summary["label"].unique():
        sub = summary[summary["label"] == label]
        plt.plot(sub["epsilon"], sub["mean_group_total_reward"], marker="o", label=label)
    plt.title("Average Group Total Reward vs Epsilon")
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
    plt.title("Average Reward Per Agent vs Epsilon")
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
    plt.title("Average Collision Rate vs Epsilon")
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
    plt.title("Average Reward Inequality vs Epsilon")
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
    plt.title("Normalized Group Reward vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Normalized Group Reward")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "normalized_group_reward_vs_epsilon.png"), dpi=300)
    plt.show()

    # 6. Best epsilon by configuration
    best = summary.loc[summary.groupby("label")["mean_group_total_reward"].idxmax()]
    best = best.sort_values(["n_agents", "n_arms"])

    plt.figure(figsize=(10, 5))
    plt.plot(best["label"], best["epsilon"], marker="o")
    plt.title("Best Epsilon by Scaling Configuration")
    plt.xlabel("Configuration")
    plt.ylabel("Best Epsilon")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_epsilon_by_configuration.png"), dpi=300)
    plt.show()

    # 7. Best epsilon vs number of agents
    plt.figure(figsize=(8, 5))
    plt.plot(best["n_agents"], best["epsilon"], marker="o")
    plt.title("Best Epsilon vs Number of Agents")
    plt.xlabel("Number of Agents")
    plt.ylabel("Best Epsilon")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_epsilon_vs_n_agents.png"), dpi=300)
    plt.show()

    print(f"Plots saved in: {out_dir}")


if __name__ == "__main__":
    main(
        csv_path=r"/mnt/data/experiment_1_to_30_agents.csv"
    )