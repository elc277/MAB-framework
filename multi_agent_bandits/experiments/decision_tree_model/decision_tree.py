import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Ensures plots are saved even if no plot window opens
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


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
    """
    Creates one row per configuration.

    For each configuration, the best epsilon is defined as the epsilon
    with the highest average group total reward across seeds.
    """

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


def save_feature_importance_plot(importance, out_dir):
    plt.figure(figsize=(10, 6))

    top_features = importance.head(15)

    plt.barh(top_features["feature"], top_features["importance"])
    plt.gca().invert_yaxis()
    plt.title("Decision Tree Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    output_path = os.path.join(out_dir, "decision_tree_feature_importance.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def save_predicted_vs_actual_plot(y_test, test_pred, y_min, y_max, out_dir):
    plt.figure(figsize=(6, 6))

    plt.scatter(y_test, test_pred, alpha=0.7)
    plt.plot([y_min, y_max], [y_min, y_max], linestyle="--")

    plt.title("Predicted vs Actual Best Epsilon")
    plt.xlabel("Actual Best Epsilon")
    plt.ylabel("Predicted Best Epsilon")
    plt.tight_layout()

    output_path = os.path.join(out_dir, "predicted_vs_actual_best_epsilon.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def save_full_tree_plot(tree, feature_names, out_dir):
    plt.figure(figsize=(28, 14))

    plot_tree(
        tree,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=8
    )

    plt.title("Decision Tree for Best Epsilon")
    plt.tight_layout()

    output_path = os.path.join(out_dir, "decision_tree_visualization_full.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def save_simplified_tree_plot(tree, feature_names, out_dir):
    """
    Saves a simpler tree figure for use in the paper.
    This does not retrain the model; it only limits the displayed depth.
    """

    plt.figure(figsize=(18, 8))

    plot_tree(
        tree,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=9,
        max_depth=2
    )

    plt.title("Simplified Decision Tree for Best Epsilon")
    plt.tight_layout()

    output_path = os.path.join(out_dir, "decision_tree_visualization_simplified.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def train_decision_tree(best_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    
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

    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    print("Decision Tree Results")
    print("---------------------")
    print(f"Train R²: {train_r2:.3f}")
    print(f"Test R²:  {test_r2:.3f}")
    print(f"Test MAE: {test_mae:.4f}")

    scores_path = os.path.join(out_dir, "decision_tree_scores.txt")
    with open(scores_path, "w", encoding="utf-8") as f:
        f.write("Decision Tree Results\n")
        f.write("---------------------\n")
        f.write(f"Train R²: {train_r2:.3f}\n")
        f.write(f"Test R²:  {test_r2:.3f}\n")
        f.write(f"Test MAE: {test_mae:.4f}\n")

    print(f"Saved: {scores_path}")

    results = best_df.copy()
    results["predicted_best_epsilon"] = tree.predict(X_encoded)

    predictions_path = os.path.join(out_dir, "best_epsilon_with_tree_predictions.csv")
    results.to_csv(predictions_path, index=False)
    print(f"Saved: {predictions_path}")

    importance = pd.DataFrame({
        "feature": X_encoded.columns,
        "importance": tree.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_path = os.path.join(out_dir, "decision_tree_feature_importance.csv")
    importance.to_csv(importance_path, index=False)
    print(f"Saved: {importance_path}")

    save_feature_importance_plot(importance, out_dir)
    save_predicted_vs_actual_plot(y_test, test_pred, y.min(), y.max(), out_dir)
    save_full_tree_plot(tree, list(X_encoded.columns), out_dir)
    save_simplified_tree_plot(tree, list(X_encoded.columns), out_dir)

    tree_rules = export_text(
        tree,
        feature_names=list(X_encoded.columns)
    )

    rules_path = os.path.join(out_dir, "decision_tree_rules.txt")
    with open(rules_path, "w", encoding="utf-8") as f:
        f.write(tree_rules)

    print(f"Saved: {rules_path}")

    print(f"\nAll outputs saved in: {out_dir}")

def main():
    csv_paths = [
        r"results/full_ratio_sweep_1_to_100_agents_full_value_collision_multicore/full_ratio_sweep_1_to_100_agents_full_value_collision_multicore.csv",
        r"results/full_ratio_sweep_1_to_100_agents_linear_collision_multicore/full_ratio_sweep_1_to_100_agents_linear_collision_multicore.csv",
        r"results/full_ratio_sweep_1_to_100_agents_random_roulette_multicore/full_ratio_sweep_1_to_100_agents_random_roulette_collision_multicore.csv",
        r"results/full_ratio_sweep_1_to_100_agents_zero_on_collision_multicore/full_ratio_sweep_1_to_100_agents_zero_on_collision_multicore.csv",
        r"results/full_ratio_sweep_1_to_100_agents_degradation_collision_multicore_part_1_under/full_ratio_sweep_degradation_complete.csv",
    ]

    out_dir = r"results/epsilon_decision_tree_model"
    os.makedirs(out_dir, exist_ok=True)

    df = load_experiment_csvs(csv_paths)

    best_df = create_best_epsilon_dataset(df)

    best_dataset_path = os.path.join(out_dir, "best_epsilon_modelling_dataset.csv")
    best_df.to_csv(best_dataset_path, index=False)

    print(f"Best epsilon modelling dataset saved to: {best_dataset_path}")
    print(f"Number of modelling rows: {len(best_df)}")

    train_decision_tree(best_df, out_dir)


if __name__ == "__main__":
    main()