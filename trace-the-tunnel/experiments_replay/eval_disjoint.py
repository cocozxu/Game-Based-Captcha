"""
Disjoint-source evaluation for the replay experiment.

The original v1 result (RF AUC = 0.996, human-vs-replay) is potentially
contaminated by source-set leakage: the classifier was trained on the
same human pool that the replay agent sampled from, so some "human"
trajectories in the CV folds had been used as replay sources in other
folds.

This script removes that contamination:

  1. Loads the allowlist file (session_ids the replay agent was
     allowed to sample from for the named experiment).
  2. Builds the feature matrix from analysis/features.py.
  3. Excludes those session_ids from the "human" class, so the eval
     humans are strictly disjoint from the replay sources.
  4. Runs LR / RF / GB with stratified 5-fold CV on
     (held-out humans vs replay_<name>) and prints AUC + per-feature
     importance.

Usage:

  python experiments_replay/eval_disjoint.py --name replay_v1b \
      --allowlist experiments_replay/allowed_sessions_v1b.json
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "analysis"))
from features import build_feature_dataframe  # noqa: E402

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


def run_pair(df, label_a, label_b, feature_cols):
    sub = df[df["source"].isin([label_a, label_b])].copy()
    sub["y"] = (sub["source"] == label_b).astype(int)
    X = sub[feature_cols].fillna(0).values
    y = sub["y"].values
    print(f"\n  {label_a} (n={(y == 0).sum()})  vs  {label_b} (n={(y == 1).sum()})")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = {}
    for cname, model in [
        ("LR", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000, random_state=42))])),
        ("RF", Pipeline([("s", StandardScaler()), ("c", RandomForestClassifier(n_estimators=200, random_state=42))])),
        ("GB", Pipeline([("s", StandardScaler()), ("c", GradientBoostingClassifier(n_estimators=200, random_state=42))])),
    ]:
        y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        auc = roc_auc_score(y, y_prob)
        acc = accuracy_score(y, y_pred)
        f1 = cross_validate(model, X, y, cv=cv, scoring=["f1"], return_train_score=False)["test_f1"].mean()
        print(f"    {cname}:  AUC={auc:.3f}  acc={acc:.3f}  f1={f1:.3f}")
        out[cname] = {"auc": float(auc), "acc": float(acc), "f1": float(f1)}
    rf = Pipeline([("s", StandardScaler()), ("c", RandomForestClassifier(n_estimators=300, random_state=42))])
    rf.fit(X, y)
    imp = sorted(zip(feature_cols, rf.named_steps["c"].feature_importances_), key=lambda t: -t[1])
    print(f"    Top-5 features:")
    for fname, v in imp[:5]:
        print(f"      {fname:<28} {v:.3f}")
    return out, imp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Replay variant name (e.g. replay_v1b).")
    parser.add_argument("--allowlist", required=True, help="JSON file with the session_ids used as replay sources.")
    parser.add_argument("--out", default=None, help="Where to write the result JSON (default: analysis/<name>_eval.json).")
    args = parser.parse_args()

    with open(args.allowlist) as f:
        source_ids = set(json.load(f))
    print(f"Source bank size: {len(source_ids)} session_ids excluded from human eval set")

    print("\nBuilding features...")
    df = build_feature_dataframe(os.path.join(REPO, "data"))
    df = df[df["completed"] == True]
    df = df[df["n_points"] >= 10]
    print(f"Completed trajectories: {len(df)}")
    print(f"  By source: {df['source'].value_counts().to_dict()}")

    # Strip source-bank humans from the eval set.
    n_before = (df["source"] == "human").sum()
    df = df[~((df["source"] == "human") & (df["session_id"].isin(source_ids)))]
    n_after = (df["source"] == "human").sum()
    print(f"  Removed {n_before - n_after} bank humans from eval (held-out human n={n_after})")

    # Sanity: confirm none of the source bank session_ids leaked into the replay set
    replay_sids = set(df[df["source"] == args.name]["session_id"])
    overlap = replay_sids & source_ids
    if overlap:
        print(f"  WARNING: {len(overlap)} replay session_ids overlap with the source bank — should be 0.")

    print("\n=== DISJOINT EVAL ===")
    r_disjoint, imp_disjoint = run_pair(df, "human", args.name, FEATURE_COLS)

    # Optional baselines, for context
    other_replays = [s for s in df["source"].unique() if s.startswith("replay") and s != args.name]
    other_agents = [s for s in df["source"].unique()
                    if s != "human" and s != args.name and not s.startswith("replay")]

    aux = {}
    if "visual" in df["source"].unique():
        print("\n=== CONTEXT: human vs visual (baseline content-tell agent) ===")
        aux["human_vs_visual"], _ = run_pair(df, "human", "visual", FEATURE_COLS)
    for o in other_replays:
        print(f"\n=== CONTEXT: human (disjoint-from-{args.name}-bank) vs {o} ===")
        aux[f"human_vs_{o}"], _ = run_pair(df, "human", o, FEATURE_COLS)
    for o in other_agents:
        print(f"\n=== CONTEXT: human (disjoint-from-{args.name}-bank) vs {o} ===")
        aux[f"human_vs_{o}"], _ = run_pair(df, "human", o, FEATURE_COLS)

    key_feats = ["centerline_dev_mean", "centerline_dev_std", "dt_mean", "dt_std",
                 "speed_mean", "speed_std", "duration_ms", "direction_changes",
                 "tremor_power_8_12hz"]
    print("\n=== PER-SOURCE FEATURE MEANS (top tells) ===")
    print(df.groupby("source")[key_feats].mean().round(3).to_string())

    out_path = args.out or os.path.join(REPO, "analysis", f"{args.name}_eval.json")
    summary = {
        "experiment": args.name,
        "allowlist": args.allowlist,
        "n_bank_session_ids": len(source_ids),
        "n_eval_humans": int(n_after),
        "n_replay_trajectories": int((df["source"] == args.name).sum()),
        "disjoint": r_disjoint,
        "context": aux,
        "top_features_disjoint": [{"feature": f, "importance": float(v)} for f, v in imp_disjoint[:10]],
        "per_source_means": df.groupby("source")[key_feats].mean().round(4).to_dict(),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
