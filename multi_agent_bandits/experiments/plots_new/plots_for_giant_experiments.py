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
        "collision_policy",
        "target_ratio",
        "target_agent_part",
        "target_arm_part",
        "n_agents",
        "n_arms",
        "actual_agent_to_arm_ratio",
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
    policies = sorted(df["collision_policy"].dropna().unique())
    if len(policies) == 1:
        policy_text = f" — {policies[0]}"

    ratio_order = [
        "20:1", "5:1", "4:1", "3:1", "2:1",
        "1:1",
        "1:2", "1:3", "1:4", "1:5", "1:20"
    ]

    summary = (
        df.groupby(
            [
                "target_ratio",
                "target_agent_part",
                "target_arm_part",
                "n_agents",
                "n_arms",
                "actual_agent_to_arm_ratio",
                "epsilon",
            ],
            as_index=False
        )
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

    summary["ratio_numeric"] = summary["target_agent_part"] / summary["target_arm_part"]

    summary["target_ratio"] = pd.Categorical(
        summary["target_ratio"],
        categories=ratio_order,
        ordered=True
    )

    summary = summary.sort_values(
        ["target_ratio", "n_agents", "n_arms", "epsilon"]
    )

    summary["label"] = summary.apply(
        lambda row: f"{row['target_ratio']} | {int(row['n_agents'])}A-{int(row['n_arms'])}K",
        axis=1
    )

    # Keep the original-style plots only for 100 agents
    plot_df = summary[summary["n_agents"] == 100].copy()
    agent_text = f" (100 agents, multiple ratios{policy_text})"

    # 1. Group reward vs epsilon
    plt.figure(figsize=(10, 6))
    for label in plot_df["label"].unique():
        sub = plot_df[plot_df["label"] == label]
        plt.plot(sub["epsilon"], sub["mean_group_total_reward"], marker="o", label=label)

    plt.title(f"Average Group Total Reward vs Epsilon{agent_text}")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Group Total Reward")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "group_reward_vs_epsilon_100_agents.png"), dpi=300)
    plt.show()

    # 2. Mean agent reward vs epsilon
    plt.figure(figsize=(10, 6))
    for label in plot_df["label"].unique():
        sub = plot_df[plot_df["label"] == label]
        plt.plot(sub["epsilon"], sub["mean_mean_agent_reward"], marker="o", label=label)

    plt.title(f"Average Reward Per Agent vs Epsilon{agent_text}")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Reward Per Agent")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_agent_reward_vs_epsilon_100_agents.png"), dpi=300)
    plt.show()

    # 3. Collision rate vs epsilon
    plt.figure(figsize=(10, 6))
    for label in plot_df["label"].unique():
        sub = plot_df[plot_df["label"] == label]
        plt.plot(sub["epsilon"], sub["mean_collision_rate"], marker="o", label=label)

    plt.title(f"Average Collision Rate vs Epsilon{agent_text}")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Collision Rate")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "collision_rate_vs_epsilon_100_agents.png"), dpi=300)
    plt.show()

    # 4. Reward inequality vs epsilon
    plt.figure(figsize=(10, 6))
    for label in plot_df["label"].unique():
        sub = plot_df[plot_df["label"] == label]
        plt.plot(sub["epsilon"], sub["mean_reward_inequality"], marker="o", label=label)

    plt.title(f"Average Reward Inequality vs Epsilon{agent_text}")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Reward Inequality")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_inequality_vs_epsilon_100_agents.png"), dpi=300)
    plt.show()

    # 5. Normalized group reward vs epsilon
    norm_df = plot_df.copy()
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
    plt.savefig(os.path.join(out_dir, "normalized_group_reward_vs_epsilon_100_agents.png"), dpi=300)
    plt.show()

    # Best epsilon table for every ratio + agent count
    best = summary.loc[
        summary.groupby(["target_ratio", "n_agents"])["mean_group_total_reward"].idxmax()
    ].copy()

    best = best.sort_values(["target_ratio", "n_agents"])

    best["config_label"] = best.apply(
        lambda row: f"{row['target_ratio']} | {int(row['n_agents'])}A-{int(row['n_arms'])}K",
        axis=1
    )

    best.to_csv(os.path.join(out_dir, "best_epsilon_summary.csv"), index=False)

    # 6. Best epsilon by configuration, only for 100 agents
    best_100 = best[best["n_agents"] == 100].copy()
    best_100 = best_100.sort_values("target_ratio")

    plt.figure(figsize=(11, 5))
    plt.plot(best_100["config_label"], best_100["epsilon"], marker="o")
    plt.title(f"Best Epsilon by Configuration{agent_text}")
    plt.xlabel("Configuration")
    plt.ylabel("Best Epsilon")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_epsilon_by_configuration_100_agents.png"), dpi=300)
    plt.show()

    # 7. Best epsilon vs ratio, only for 100 agents
    plt.figure(figsize=(9, 5))
    plt.plot(best_100["target_ratio"].astype(str), best_100["epsilon"], marker="o")
    plt.title(f"Best Epsilon vs Ratio at 100 Agents{policy_text}")
    plt.xlabel("Target Ratio (Agents:Arms)")
    plt.ylabel("Best Epsilon")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_epsilon_vs_ratio_100_agents.png"), dpi=300)
    plt.show()

    # 8. Heatmap: best epsilon by ratio and number of agents
    heatmap_data = best.pivot(
        index="target_ratio",
        columns="n_agents",
        values="epsilon"
    )

    heatmap_data = heatmap_data.reindex(ratio_order)

    plt.figure(figsize=(14, 5))
    plt.imshow(heatmap_data, aspect="auto", origin="lower")
    plt.colorbar(label="Best Epsilon")

    plt.yticks(
        ticks=range(len(heatmap_data.index)),
        labels=heatmap_data.index
    )

    x_ticks = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    x_labels = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    plt.xticks(ticks=x_ticks, labels=x_labels)

    plt.title(f"Best Epsilon Heatmap by Ratio and Number of Agents{policy_text}")
    plt.xlabel("Number of Agents")
    plt.ylabel("Target Ratio (Agents:Arms)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_epsilon_heatmap_by_ratio_and_agents.png"), dpi=300)
    plt.show()

    # 9. Best epsilon vs number of agents, split into ratio groups
    ratio_groups = {
        "Crowded ratios": ["20:1", "5:1", "4:1", "3:1", "2:1"],
        "Balanced ratio": ["1:1"],
        "Arm-rich ratios": ["1:2", "1:3", "1:4", "1:5", "1:20"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (group_name, ratios) in zip(axes, ratio_groups.items()):
        for ratio in ratios:
            sub = best[best["target_ratio"].astype(str) == ratio]
            if not sub.empty:
                ax.plot(sub["n_agents"], sub["epsilon"], marker="o", label=ratio)

        ax.set_title(group_name)
        ax.set_xlabel("Number of Agents")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Best Epsilon")
    fig.suptitle(f"Best Epsilon vs Number of Agents, Split by Ratio Group{policy_text}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_epsilon_vs_agents_split_by_ratio_group.png"), dpi=300)
    plt.show()

    print(f"Plots saved in: {out_dir}")
    print(f"Best epsilon summary saved to: {os.path.join(out_dir, 'best_epsilon_summary.csv')}")


if __name__ == "__main__":
    main(
        csv_path=r"results/full_ratio_sweep_1_to_100_agents_zero_on_collision_multicore/full_ratio_sweep_1_to_100_agents_zero_on_collision_multicore.csv",
        out_dir=r"multi_agent_bandits/experiments/plots_new/full_ratio_sweep_1_to_100_agents_zero_on_collision_multicore_plots"
    )