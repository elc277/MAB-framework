import os
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("Agg")

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# ============================================================
# Global paths
# ============================================================

CSV_PATHS = {
    "Full reward": r"results/full_ratio_sweep_1_to_100_agents_full_value_collision_multicore/full_ratio_sweep_1_to_100_agents_full_value_collision_multicore.csv",
    "Linear share": r"results/full_ratio_sweep_1_to_100_agents_linear_collision_multicore/full_ratio_sweep_1_to_100_agents_linear_collision_multicore.csv",
    "Random roulette": r"results/full_ratio_sweep_1_to_100_agents_random_roulette_multicore/full_ratio_sweep_1_to_100_agents_random_roulette_collision_multicore.csv",
    "Zero-on-collision": r"results/full_ratio_sweep_1_to_100_agents_zero_on_collision_multicore/full_ratio_sweep_1_to_100_agents_zero_on_collision_multicore.csv",
    "Degradation": r"results/full_ratio_sweep_1_to_100_agents_degradation_collision_multicore_part_1_under/full_ratio_sweep_degradation_complete.csv",
}

OUT_DIR = r"results/poster_plots_large_labels"

RATIO_ORDER = [
    "20:1", "5:1", "4:1", "3:1", "2:1",
    "1:1",
    "1:2", "1:3", "1:4", "1:5", "1:20"
]


# ============================================================
# Poster style settings
# ============================================================

TITLE_SIZE = 20
AXIS_LABEL_SIZE = 18
TICK_SIZE = 15
LEGEND_SIZE = 15
COLORBAR_LABEL_SIZE = 17
COLORBAR_TICK_SIZE = 14

LINE_WIDTH = 3
MARKER_SIZE = 9


def apply_bold_ticks(ax):
    """Make tick labels larger and bold."""
    for tick in ax.get_xticklabels():
        tick.set_fontsize(TICK_SIZE)
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontsize(TICK_SIZE)
        tick.set_fontweight("bold")


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Shared helper functions
# ============================================================

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
        .agg(mean_group_total_reward=("group_total_reward", "mean"))
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
            raise FileNotFoundError(f"CSV not found for {policy_name}: {csv_path}")

        df = pd.read_csv(csv_path)
        best = compute_best_epsilon(df)
        best["policy_label"] = policy_name
        all_best.append(best)

    return pd.concat(all_best, ignore_index=True)


def load_policy_best_epsilon(policy_name):
    csv_path = CSV_PATHS[policy_name]

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found for {policy_name}: {csv_path}")

    df = pd.read_csv(csv_path)
    best = compute_best_epsilon(df)
    best["policy_label"] = policy_name
    return best


# ============================================================
# Plot 1: Best epsilon vs ratio at 100 agents, all policies
# ============================================================

def plot_all_policies_best_epsilon_100_agents():
    best_all = load_all_best_epsilons()

    best_100 = best_all[best_all["n_agents"] == 100].copy()
    best_100["target_ratio"] = pd.Categorical(
        best_100["target_ratio"],
        categories=RATIO_ORDER,
        ordered=True
    )
    best_100 = best_100.sort_values(["target_ratio", "policy_label"])

    fig, ax = plt.subplots(figsize=(10, 6))

    for policy in ["Degradation", "Full reward", "Linear share", "Random roulette", "Zero-on-collision"]:
        sub = best_100[best_100["policy_label"] == policy].sort_values("target_ratio")
        ax.plot(
            sub["target_ratio"].astype(str),
            sub["epsilon"],
            marker="o",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=policy
        )

    ax.set_title(
        "Best Epsilon vs Ratio at 100 Agents",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )
    ax.set_xlabel(
        "Target Ratio (Agents:Arms)",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold"
    )
    ax.set_ylabel(
        "Best Epsilon",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold"
    )

    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")

    apply_bold_ticks(ax)

    legend = ax.legend(fontsize=LEGEND_SIZE, frameon=True)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.tight_layout()

    output_path = os.path.join(
        OUT_DIR,
        "poster_combined_best_epsilon_vs_ratio_100_agents_all_policies.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# Plot 2 and 3: Heatmaps for degradation and linear share
# ============================================================

def plot_policy_heatmap(policy_name, output_filename):
    best = load_policy_best_epsilon(policy_name)

    heatmap_data = best.pivot(
        index="target_ratio",
        columns="n_agents",
        values="epsilon"
    )

    heatmap_data = heatmap_data.reindex(RATIO_ORDER)

    fig, ax = plt.subplots(figsize=(14, 5))

    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        origin="lower",
        vmin=0.01,
        vmax=0.50
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        "Best Epsilon",
        fontsize=COLORBAR_LABEL_SIZE,
        fontweight="bold"
    )
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)

    x_ticks = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    x_labels = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    ax.set_xticks(ticks=x_ticks)
    ax.set_xticklabels(x_labels)

    ax.set_title(
        f"Best Epsilon Heatmap by Ratio and Number of Agents — {policy_name}",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )
    ax.set_xlabel(
        "Number of Agents",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold"
    )
    ax.set_ylabel(
        "Target Ratio (Agents:Arms)",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold"
    )

    apply_bold_ticks(ax)

    fig.tight_layout()

    output_path = os.path.join(OUT_DIR, output_filename)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# Plot 4: Best epsilon vs ratio, linear share vs random roulette
# ============================================================

def plot_linear_vs_random_roulette_best_epsilon_100_agents():
    best_all = load_all_best_epsilons()

    best_100 = best_all[best_all["n_agents"] == 100].copy()
    best_100["target_ratio"] = pd.Categorical(
        best_100["target_ratio"],
        categories=RATIO_ORDER,
        ordered=True
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for policy in ["Linear share", "Random roulette"]:
        sub = best_100[best_100["policy_label"] == policy].sort_values("target_ratio")
        ax.plot(
            sub["target_ratio"].astype(str),
            sub["epsilon"],
            marker="o",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=policy
        )

    ax.set_title(
        "Best Epsilon vs Ratio: Linear Share vs Random Roulette",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )
    ax.set_xlabel(
        "Target Ratio (Agents:Arms)",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold"
    )
    ax.set_ylabel(
        "Best Epsilon",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold"
    )

    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")

    apply_bold_ticks(ax)

    legend = ax.legend(fontsize=LEGEND_SIZE, frameon=True)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.tight_layout()

    output_path = os.path.join(
        OUT_DIR,
        "poster_best_epsilon_linear_vs_random_roulette_100_agents.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# Plot 5: Decision tree feature importance
# ============================================================

def load_experiment_csvs(csv_paths):
    dfs = []

    for path in csv_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path)

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
            raise ValueError(f"Missing columns in {path}: {missing}")

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def create_best_epsilon_dataset(df):
    summary = (
        df.groupby(
            [
                "collision_policy",
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
            mean_agent_reward=("mean_agent_reward", "mean"),
            mean_collision_rate=("collision_rate", "mean"),
            mean_reward_inequality=("reward_inequality", "mean"),
        )
    )

    best = summary.loc[
        summary.groupby(
            [
                "collision_policy",
                "target_ratio",
                "target_agent_part",
                "target_arm_part",
                "n_agents",
                "n_arms",
                "actual_agent_to_arm_ratio",
            ]
        )["mean_group_total_reward"].idxmax()
    ].copy()

    best = best.rename(columns={"epsilon": "best_epsilon"})
    best["agent_to_arm_ratio"] = best["actual_agent_to_arm_ratio"]
    best["arms_per_agent"] = best["n_arms"] / best["n_agents"]

    return best


def plot_decision_tree_feature_importance():
    csv_paths = list(CSV_PATHS.values())

    df = load_experiment_csvs(csv_paths)
    best_df = create_best_epsilon_dataset(df)

    feature_columns = [
        "collision_policy",
        "n_agents",
        "agent_to_arm_ratio",
    ]

    X = best_df[feature_columns].copy()
    y = best_df["best_epsilon"].copy()

    X_encoded = pd.get_dummies(
        X,
        columns=["collision_policy"],
        drop_first=False
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.25,
        random_state=42
    )

    tree = DecisionTreeRegressor(
        max_depth=4,
        min_samples_leaf=10,
        random_state=42
    )

    tree.fit(X_train, y_train)

    test_pred = tree.predict(X_test)
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    print(f"Decision tree test R²: {test_r2:.3f}")
    print(f"Decision tree test MAE: {test_mae:.4f}")

    importance = pd.DataFrame({
        "feature": X_encoded.columns,
        "importance": tree.feature_importances_
    }).sort_values("importance", ascending=False)

    top_features = importance.head(15)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        top_features["feature"],
        top_features["importance"]
    )

    ax.invert_yaxis()

    ax.set_title(
        "Decision Tree Feature Importance",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )
    ax.set_xlabel(
        "Relative Importance Score",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold"
    )
    ax.set_ylabel(
        "Feature",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold"
    )

    apply_bold_ticks(ax)

    fig.tight_layout()

    output_path = os.path.join(
        OUT_DIR,
        "poster_decision_tree_feature_importance.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    ensure_out_dir()

    plot_all_policies_best_epsilon_100_agents()

    plot_policy_heatmap(
        policy_name="Degradation",
        output_filename="poster_best_epsilon_heatmap_degradation.png"
    )

    plot_policy_heatmap(
        policy_name="Linear share",
        output_filename="poster_best_epsilon_heatmap_linear_share.png"
    )

    plot_linear_vs_random_roulette_best_epsilon_100_agents()

    plot_decision_tree_feature_importance()

    print(f"\nAll poster plots saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()