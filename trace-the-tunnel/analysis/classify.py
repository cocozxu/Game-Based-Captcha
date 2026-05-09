"""
Train and evaluate classifiers to distinguish human vs agent trajectories.

Run: python analysis/classify.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

from features import build_feature_dataframe


BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__))

# Optional: matplotlib for plots
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ---------------------------------------------------------------------------
# Feature columns used for classification
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "speed_mean", "speed_std", "speed_max", "speed_median",
    "accel_mean", "accel_std", "accel_max",
    "jerk_mean", "jerk_std", "jerk_max",
    "curvature_mean", "curvature_std", "curvature_max",
    "dt_mean", "dt_std", "dt_min", "dt_max",
    "path_length", "path_efficiency", "direction_changes",
    "centerline_dev_mean", "centerline_dev_std",
    "duration_ms", "n_points",
    "tremor_power_8_12hz",
]


def prepare_data(df, success_only=True):
    """Filter and prepare feature matrix X and label vector y."""
    if success_only:
        df = df[df["completed"] == True].copy()

    # Only keep rows with sufficient data
    df = df[df["n_points"] >= 10].copy()

    # Use available feature columns
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0).values
    y = df["is_agent"].values
    sources = df["source"].values

    return X, y, sources, available, df


def evaluate_classifiers(X, y, sources, feature_names):
    """Run multiple classifiers with stratified cross-validation."""
    cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y))), shuffle=True, random_state=42)

    classifiers = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]),
    }

    results = {}
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS (Human=0 vs Agent=1)")
    print("=" * 60)

    for name, pipeline in classifiers.items():
        print(f"\n--- {name} ---")

        # Cross-validated predictions
        y_pred = cross_val_predict(pipeline, X, y, cv=cv)

        # Cross-validated scores
        scores = cross_validate(pipeline, X, y, cv=cv,
                                scoring=["accuracy", "f1", "roc_auc"],
                                return_train_score=True)

        acc = accuracy_score(y, y_pred)
        try:
            y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, y_prob)
        except Exception:
            auc = scores["test_roc_auc"].mean()

        print(f"Accuracy:  {acc:.3f}")
        print(f"ROC AUC:   {auc:.3f}")
        print(f"CV Acc:    {scores['test_accuracy'].mean():.3f} +/- {scores['test_accuracy'].std():.3f}")
        print(f"CV F1:     {scores['test_f1'].mean():.3f} +/- {scores['test_f1'].std():.3f}")
        print()
        print(classification_report(y, y_pred, target_names=["human", "agent"]))
        print("Confusion matrix:")
        print(confusion_matrix(y, y_pred))

        results[name] = {
            "accuracy": acc,
            "auc": auc,
            "cv_accuracy": scores["test_accuracy"].tolist(),
            "cv_f1": scores["test_f1"].tolist(),
        }

    return results


def feature_importance(X, y, feature_names):
    """Train a Random Forest and report feature importances."""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("=" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_scaled, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n{'Rank':<6}{'Feature':<30}{'Importance':<12}")
    print("-" * 48)
    for rank, idx in enumerate(indices, 1):
        print(f"{rank:<6}{feature_names[idx]:<30}{importances[idx]:<12.4f}")

    if HAS_PLT:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_n = min(15, len(indices))
        top_idx = indices[:top_n]
        ax.barh(range(top_n),
                importances[top_idx],
                color="#4fc3f7", alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in top_idx])
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Top Feature Importances (Random Forest)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "plots", "feature_importance.png"))
        plt.close(fig)
        print(f"\n  Saved plots/feature_importance.png")

    return dict(zip(feature_names, importances.tolist()))


def per_source_breakdown(df):
    """Show stats per source type."""
    print("\n" + "=" * 60)
    print("PER-SOURCE SUMMARY")
    print("=" * 60)

    for src in sorted(df["source"].unique()):
        sub = df[df["source"] == src]
        completed = sub[sub["completed"] == True]
        print(f"\n  {src}:")
        print(f"    Total trajectories: {len(sub)}")
        print(f"    Completed:          {len(completed)}")
        if len(completed) > 0:
            print(f"    Avg duration (ms):  {completed['duration_ms'].mean():.0f}")
            print(f"    Avg n_points:       {completed['n_points'].mean():.0f}")
            print(f"    Avg speed_mean:     {completed['speed_mean'].mean():.4f}")
            print(f"    Avg dt_mean (ms):   {completed['dt_mean'].mean():.2f}")
            print(f"    Avg centerline_dev: {completed['centerline_dev_mean'].mean():.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading features...")
    df = build_feature_dataframe(BASE_DIR)
    print(f"Total trajectories: {len(df)}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")

    per_source_breakdown(df)

    # Prepare data (success only by default)
    X, y, sources, feature_names, filtered_df = prepare_data(df, success_only=True)
    print(f"\nUsing {len(X)} successful trajectories for classification")
    print(f"  Human: {(y == 0).sum()}, Agent: {(y == 1).sum()}")
    print(f"  Features: {len(feature_names)}")

    if len(np.unique(y)) < 2:
        print("\nERROR: Need both human and agent data to classify. Collect more data!")
        sys.exit(1)

    if min(np.bincount(y)) < 2:
        print("\nWARNING: Very few samples in one class. Results will be unreliable.")

    # Run classification
    results = evaluate_classifiers(X, y, sources, feature_names)

    # Feature importance
    os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)
    importances = feature_importance(X, y, feature_names)

    # Save results summary
    import json
    summary = {
        "n_trajectories": len(X),
        "n_human": int((y == 0).sum()),
        "n_agent": int((y == 1).sum()),
        "features_used": feature_names,
        "classifier_results": results,
        "feature_importances": importances,
    }
    summary_path = os.path.join(OUT_DIR, "results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {summary_path}")
