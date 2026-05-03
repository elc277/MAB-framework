import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = os.path.join("results", "pilot", "pilot_summary_with_std.csv")
    out_dir = os.path.join("results", "pilot", "plots_with_std")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    summary = (
        df.groupby(["std", "n_agents", "epsilon"], as_index=False)
        .agg(
            mean_group_total_reward=("group_total_reward", "mean"),
            std_group_total_reward=("group_total_reward", "std"),
            mean_collision_rate=("collision_rate", "mean"),
            std_collision_rate=("collision_rate", "std"),
            mean_reward_inequality=("reward_inequality", "mean"),
            std_reward_inequality=("reward_inequality", "std"),
        )
    )

    summary = summary.sort_values(["std", "n_agents", "epsilon"])

    std_values = sorted(summary["std"].unique())

    for std in std_values:
        std_data = summary[summary["std"] == std]

        # Plot 1: Group reward vs epsilon
        plt.figure(figsize=(8, 5))
        for n_agents in sorted(std_data["n_agents"].unique()):
            sub = std_data[std_data["n_agents"] == n_agents]
            plt.plot(
                sub["epsilon"],
                sub["mean_group_total_reward"],
                marker="o",
                label=f"{n_agents} agents"
            )

        plt.title(f"Average Group Total Reward vs Epsilon (std = {std})")
        plt.xlabel("Epsilon")
        plt.ylabel("Average Group Total Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"group_reward_vs_epsilon_std_{std}.png"), dpi=300)
        plt.show()

        # Plot 2: Collision rate vs epsilon
        plt.figure(figsize=(8, 5))
        for n_agents in sorted(std_data["n_agents"].unique()):
            sub = std_data[std_data["n_agents"] == n_agents]
            plt.plot(
                sub["epsilon"],
                sub["mean_collision_rate"],
                marker="o",
                label=f"{n_agents} agents"
            )

        plt.title(f"Average Collision Rate vs Epsilon (std = {std})")
        plt.xlabel("Epsilon")
        plt.ylabel("Average Collision Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"collision_rate_vs_epsilon_std_{std}.png"), dpi=300)
        plt.show()

        # Plot 3: Reward inequality vs epsilon
        plt.figure(figsize=(8, 5))
        for n_agents in sorted(std_data["n_agents"].unique()):
            sub = std_data[std_data["n_agents"] == n_agents]
            plt.plot(
                sub["epsilon"],
                sub["mean_reward_inequality"],
                marker="o",
                label=f"{n_agents} agents"
            )

        plt.title(f"Average Reward Inequality vs Epsilon (std = {std})")
        plt.xlabel("Epsilon")
        plt.ylabel("Average Reward Inequality")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"reward_inequality_vs_epsilon_std_{std}.png"), dpi=300)
        plt.show()

        # Plot 4: Heatmap of group reward
        pivot_reward = std_data.pivot(index="n_agents", columns="epsilon", values="mean_group_total_reward")

        plt.figure(figsize=(8, 4))
        plt.imshow(pivot_reward, aspect="auto")
        plt.colorbar(label="Average Group Total Reward")
        plt.title(f"Reward Heatmap by Agent Count and Epsilon (std = {std})")
        plt.xlabel("Epsilon")
        plt.ylabel("Number of Agents")
        plt.xticks(range(len(pivot_reward.columns)), [f"{x:.2f}" for x in pivot_reward.columns])
        plt.yticks(range(len(pivot_reward.index)), pivot_reward.index)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"reward_heatmap_std_{std}.png"), dpi=300)
        plt.show()

    # Extra plot: compare reward by std for each agent count
    plt.figure(figsize=(9, 6))
    for n_agents in sorted(summary["n_agents"].unique()):
        sub = summary[summary["n_agents"] == n_agents]

        # For each std, get the best reward across epsilons
        best_per_std = (
            sub.groupby("std", as_index=False)["mean_group_total_reward"]
            .max()
            .sort_values("std")
        )

        plt.plot(
            best_per_std["std"],
            best_per_std["mean_group_total_reward"],
            marker="o",
            label=f"{n_agents} agents"
        )

    plt.title("Best Average Group Reward vs Reward Std")
    plt.xlabel("Arm Standard Deviation")
    plt.ylabel("Best Average Group Total Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_group_reward_vs_std.png"), dpi=300)
    plt.show()

    print(f"Plots saved in: {out_dir}")


if __name__ == "__main__":
    main()