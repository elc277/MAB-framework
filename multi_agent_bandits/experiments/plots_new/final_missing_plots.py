import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Paths

CSV_PATHS = {
    "Full reward": r"results/full_ratio_sweep_1_to_100_agents_full_value_collision_multicore/full_ratio_sweep_1_to_100_agents_full_value_collision_multicore.csv",
    "Linear share": r"results/full_ratio_sweep_1_to_100_agents_linear_collision_multicore/full_ratio_sweep_1_to_100_agents_linear_collision_multicore.csv",
    "Random roulette": r"results/full_ratio_sweep_1_to_100_agents_random_roulette_multicore/full_ratio_sweep_1_to_100_agents_random_roulette_collision_multicore.csv",
    "Zero-on-collision": r"results/full_ratio_sweep_1_to_100_agents_zero_on_collision_multicore/full_ratio_sweep_1_to_100_agents_zero_on_collision_multicore.csv",
    "Degradation": r"results/full_ratio_sweep_1_to_100_agents_degradation_collision_multicore_part_1_under/full_ratio_sweep_degradation_complete.csv",
}

OUT_DIR = r"multi_agent_bandits/experiments/plots_new/paper_extra_visuals"

RATIO_ORDER = [
    "20:1", "5:1", "4:1", "3:1", "2:1",
    "1:1",
    "1:2", "1:3", "1:4", "1:5", "1:20"
]


# Helpers

def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def compute_best_epsilon(df):
    required_columns = [
        "target_ratio",
        "target_agent_part",
        "target_arm_part",
        "n_agents",
        "n_arms",
        "epsilon",
        "group_total_reward",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    summary = (
        df.groupby(
            [
                "target_ratio",
                "target_agent_part",
                "target_arm_part",
                "n_agents",
                "n_arms",
                "epsilon",
            ],
            as_index=False
        )
        .agg(
            mean_group_total_reward=("group_total_reward", "mean")
        )
    )

    best = summary.loc[
        summary.groupby(["target_ratio", "n_agents"])["mean_group_total_reward"].idxmax()
    ].copy()

    best["target_ratio"] = pd.Categorical(
        best["target_ratio"],
        categories=RATIO_ORDER,
        ordered=True
    )

    best = best.sort_values(["target_ratio", "n_agents"])
    return best


def load_all_best_epsilons():
    all_best = []

    for policy_name, csv_path in CSV_PATHS.items():
        if not os.path.exists(csv_path):
            print(f"WARNING: CSV not found for {policy_name}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        best = compute_best_epsilon(df)
        best["policy_label"] = policy_name
        all_best.append(best)

    if not all_best:
        raise ValueError("No CSV files were loaded. Check the CSV paths.")

    return pd.concat(all_best, ignore_index=True)


# 1. Schematic diagram


def create_schematic_diagram():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")

    agent_positions = [(0.1, 0.8), (0.1, 0.6), (0.1, 0.4), (0.1, 0.2)]
    arm_positions = [(0.75, 0.75), (0.75, 0.5), (0.75, 0.25)]

    agent_labels = ["Agent 1", "Agent 2", "Agent 3", "Agent 4"]
    arm_labels = ["Arm A", "Arm B", "Arm C"]

    for (x, y), label in zip(agent_positions, agent_labels):
        circle = plt.Circle((x, y), 0.06, fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=9)

    for (x, y), label in zip(arm_positions, arm_labels):
        rect = plt.Rectangle((x - 0.08, y - 0.05), 0.16, 0.10, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=10)

    # Arrows: two agents choose same arm, creating collision
    arrows = [
        (agent_positions[0], arm_positions[0]),
        (agent_positions[1], arm_positions[1]),
        (agent_positions[2], arm_positions[1]),
        (agent_positions[3], arm_positions[2]),
    ]

    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", linewidth=1.8)
        )

    ax.text(
        0.75,
        0.62,
        "Collision:\nAgent 2 and Agent 3\nchoose the same arm",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round", fill=False)
    )

    ax.set_title("Multi-Agent Bandit Environment with Collisions", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "schematic_multi_agent_bandit_collision.png"), dpi=300)
    plt.show()


# 2. Ratio example table


def create_ratio_example_table():
    rows = []

    for ratio in RATIO_ORDER:
        agent_part, arm_part = map(int, ratio.split(":"))
        n_agents = 100
        n_arms = max(1, int(round(n_agents * arm_part / agent_part)))

        rows.append({
            "Target ratio (Agents:Arms)": ratio,
            "Example with 100 agents": f"{n_agents}A / {n_arms}K",
            "Interpretation": (
                "Crowded" if agent_part > arm_part
                else "Balanced" if agent_part == arm_part
                else "Arm-rich"
            )
        })

    table_df = pd.DataFrame(rows)
    table_df.to_csv(os.path.join(OUT_DIR, "ratio_example_table.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    ax.set_title("Example Configurations for Agent-to-Arm Ratios", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ratio_example_table.png"), dpi=300)
    plt.show()


# 3. Collision policy reward table

def create_collision_policy_reward_table():
    rows = []

    for m in [1, 2, 3, 4]:
        rows.append({
            "Colliding agents (m)": m,
            "Full reward": "r",
            "Linear share": f"r/{m}",
            "Random roulette": "r" if m == 1 else f"1/{m}: r, {m-1}/{m}: 0",
            "Zero-on-collision": "r" if m == 1 else "0",
            "Degradation (d=0.5)": f"r/{m * (1 + 0.5 * (m - 1)):.1f}"
        })

    table_df = pd.DataFrame(rows)
    table_df.to_csv(os.path.join(OUT_DIR, "collision_policy_reward_table.csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    ax.set_title("Reward Received per Agent under Each Collision Policy", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "collision_policy_reward_table.png"), dpi=300)
    plt.show()


# 5. Ratio range orientation figure

def create_ratio_range_figure():
    x = np.arange(len(RATIO_ORDER))

    fig, ax = plt.subplots(figsize=(11, 2.8))

    ax.plot(x, np.zeros_like(x), marker="o", linewidth=2)
    ax.axvline(RATIO_ORDER.index("1:1"), linestyle="--", linewidth=1)

    for i, ratio in enumerate(RATIO_ORDER):
        ax.text(i, 0.08, ratio, ha="center", va="bottom", rotation=45, fontsize=9)

    ax.text(1.5, -0.18, "Crowded\nmore agents than arms", ha="center", va="top", fontsize=10)
    ax.text(RATIO_ORDER.index("1:1"), -0.18, "Balanced", ha="center", va="top", fontsize=10)
    ax.text(8.5, -0.18, "Arm-rich\nmore arms than agents", ha="center", va="top", fontsize=10)

    ax.set_ylim(-0.35, 0.35)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Tested Agent-to-Arm Ratio Range", fontsize=13)
    ax.spines[["left", "right", "top", "bottom"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tested_ratio_range_orientation.png"), dpi=300)
    plt.show()


# 7, 8, 12, 14. Best epsilon comparison plots

def create_best_epsilon_comparison_plots():
    best_all = load_all_best_epsilons()

    best_100 = best_all[best_all["n_agents"] == 100].copy()
    best_100["target_ratio"] = pd.Categorical(
        best_100["target_ratio"],
        categories=RATIO_ORDER,
        ordered=True
    )
    best_100 = best_100.sort_values(["target_ratio", "policy_label"])

    best_100.to_csv(os.path.join(OUT_DIR, "best_epsilon_100_agents_all_policies.csv"), index=False)

    # 12. Combined best epsilon vs ratio, all policies
    plt.figure(figsize=(10, 6))
    for policy in best_100["policy_label"].unique():
        sub = best_100[best_100["policy_label"] == policy].sort_values("target_ratio")
        plt.plot(sub["target_ratio"].astype(str), sub["epsilon"], marker="o", label=policy)

    plt.title("Best Epsilon vs Ratio at 100 Agents")
    plt.xlabel("Target Ratio (Agents:Arms)")
    plt.ylabel("Best Epsilon")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "combined_best_epsilon_vs_ratio_100_agents_all_policies.png"), dpi=300)
    plt.show()

    # 7. Full reward vs linear share
    plt.figure(figsize=(9, 5))
    for policy in ["Full reward", "Linear share"]:
        sub = best_100[best_100["policy_label"] == policy].sort_values("target_ratio")
        plt.plot(sub["target_ratio"].astype(str), sub["epsilon"], marker="o", label=policy)

    plt.title("Best Epsilon vs Ratio: Full Reward vs Linear Share at 100 Agents")
    plt.xlabel("Target Ratio (Agents:Arms)")
    plt.ylabel("Best Epsilon")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "best_epsilon_full_reward_vs_linear_100_agents.png"), dpi=300)
    plt.show()

    # 8 / 14. Linear share vs random roulette
    plt.figure(figsize=(9, 5))
    for policy in ["Linear share", "Random roulette"]:
        sub = best_100[best_100["policy_label"] == policy].sort_values("target_ratio")
        plt.plot(sub["target_ratio"].astype(str), sub["epsilon"], marker="o", label=policy)

    plt.title("Best Epsilon vs Ratio: Linear Share vs Random Roulette at 100 Agents")
    plt.xlabel("Target Ratio (Agents:Arms)")
    plt.ylabel("Best Epsilon")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "best_epsilon_linear_vs_random_roulette_100_agents.png"), dpi=300)
    plt.show()


# Optional: combined heatmaps by policy

def create_combined_policy_heatmaps():
    best_all = load_all_best_epsilons()

    policies = [
        "Full reward",
        "Linear share",
        "Random roulette",
        "Zero-on-collision",
        "Degradation"
    ]

    fig, axes = plt.subplots(1, len(policies), figsize=(22, 5), sharey=True)

    for ax, policy in zip(axes, policies):
        sub = best_all[best_all["policy_label"] == policy].copy()

        if sub.empty:
            ax.set_title(f"{policy}\nmissing")
            ax.axis("off")
            continue

        heatmap_data = sub.pivot(
            index="target_ratio",
            columns="n_agents",
            values="epsilon"
        )

        heatmap_data = heatmap_data.reindex(RATIO_ORDER)

        im = ax.imshow(heatmap_data, aspect="auto", origin="lower", vmin=0.01, vmax=0.50)

        ax.set_title(policy)
        ax.set_xlabel("Agents")

        x_ticks = [0, 9, 19, 49, 99]
        x_labels = [1, 10, 20, 50, 100]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

        ax.set_yticks(range(len(RATIO_ORDER)))
        ax.set_yticklabels(RATIO_ORDER)

    axes[0].set_ylabel("Target Ratio (Agents:Arms)")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Best Epsilon")

    fig.suptitle("Best Epsilon Heatmaps by Ratio and Number of Agents", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "combined_best_epsilon_heatmaps_all_policies.png"), dpi=300)
    plt.show()


# Run all


def main():
    ensure_out_dir()

    create_schematic_diagram()
    create_ratio_example_table()
    create_collision_policy_reward_table()
    create_ratio_range_figure()

    create_best_epsilon_comparison_plots()
    create_combined_policy_heatmaps()

    print(f"All extra visuals saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()