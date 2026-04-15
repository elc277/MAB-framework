import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = os.path.join(
        "results",
        "experiment_12-04-2026",
        "experiment_12-04-2026.csv"
    )

    out_dir = os.path.join("results", "experiment_12-04-2026", "plots")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

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
        lambda row: f"{int(row['n_agents'])} agent(s), {int(row['n_arms'])} arm(s)",
        axis=1
    )

    # Plot 1: Group total reward vs epsilon
    plt.figure(figsize=(8, 5))
    for label in summary["label"].unique():
        sub = summary[summary["label"] == label]
        plt.plot(
            sub["epsilon"],
            sub["mean_group_total_reward"],
            marker="o",
            label=label
        )

    plt.title("Average Group Total Reward vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Group Total Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "group_reward_vs_epsilon.png"), dpi=300)
    plt.show()

    # Plot 2: Average reward per agent vs epsilon
    plt.figure(figsize=(8, 5))
    for label in summary["label"].unique():
        sub = summary[summary["label"] == label]
        plt.plot(
            sub["epsilon"],
            sub["mean_mean_agent_reward"],
            marker="o",
            label=label
        )

    plt.title("Average Reward Per Agent vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Reward Per Agent")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_agent_reward_vs_epsilon.png"), dpi=300)
    plt.show()

    # Plot 3: Collision rate vs epsilon
    plt.figure(figsize=(8, 5))
    for label in summary["label"].unique():
        sub = summary[summary["label"] == label]
        plt.plot(
            sub["epsilon"],
            sub["mean_collision_rate"],
            marker="o",
            label=label
        )

    plt.title("Average Collision Rate vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Collision Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "collision_rate_vs_epsilon.png"), dpi=300)
    plt.show()

    # Plot 4: Reward inequality vs epsilon
    plt.figure(figsize=(8, 5))
    for label in summary["label"].unique():
        sub = summary[summary["label"] == label]
        plt.plot(
            sub["epsilon"],
            sub["mean_reward_inequality"],
            marker="o",
            label=label
        )

    plt.title("Average Reward Inequality vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Reward Inequality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_inequality_vs_epsilon.png"), dpi=300)
    plt.show()

    # Plot 5: Normalized group reward vs epsilon
    norm_df = summary.copy()
    norm_df["normalized_group_reward"] = (
        norm_df.groupby("label")["mean_group_total_reward"]
        .transform(lambda x: x / x.max())
    )

    plt.figure(figsize=(8, 5))
    for label in norm_df["label"].unique():
        sub = norm_df[norm_df["label"] == label]
        plt.plot(
            sub["epsilon"],
            sub["normalized_group_reward"],
            marker="o",
            label=label
        )

    plt.title("Normalized Group Reward vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Normalized Group Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "normalized_group_reward_vs_epsilon.png"), dpi=300)
    plt.show()

    # Plot 6: Best epsilon by configuration
    best = summary.loc[summary.groupby("label")["mean_group_total_reward"].idxmax()]

    plt.figure(figsize=(7, 4))
    plt.bar(best["label"], best["epsilon"])
    plt.title("Best Epsilon by Configuration")
    plt.xlabel("Configuration")
    plt.ylabel("Best Epsilon")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_epsilon_by_configuration.png"), dpi=300)
    plt.show()

    print(f"Plots saved in: {out_dir}")


if __name__ == "__main__":
    main()