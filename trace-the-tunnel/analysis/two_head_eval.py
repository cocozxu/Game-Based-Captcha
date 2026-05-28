"""
Two-head decomposition of the human-vs-agent classifier.

For each attack source, train three RFs on (disjoint humans vs attack):
  - MOTOR head:     trajectory shape (speed/accel/jerk/curvature/path/centerline/tremor/duration)
  - MECHANISM head: how events arrive (dt_*, n_points)
  - COMBINED:       all features (the headline RF the writeup has been quoting)

Reports CV-AUC + std for each (attack, head). The motor-vs-mechanism AUC pair is
the deliverable: one point per attack in the Pareto frontier scatter plot, which
the writeup uses to claim "every attack must pick a side."

Disjointness: every human session_id that appears in any known attack allowlist
is removed from the human pool, so all attacks are evaluated against the same
held-out human eval set (uniform comparison across attacks).

Run:
  python analysis/two_head_eval.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from features import build_feature_dataframe  # noqa: E402


# ---------------------------------------------------------------------------
# Feature partition
# ---------------------------------------------------------------------------

# Mechanism head: features about HOW events are sampled / emitted.
MECHANISM_COLS = [
    "dt_mean", "dt_std", "dt_min", "dt_max",
    "n_points",
]

# Motor head: features about the PHYSICAL trajectory (shape + kinematics + spectral).
MOTOR_COLS = [
    "speed_mean", "speed_std", "speed_max", "speed_median",
    "accel_mean", "accel_std", "accel_max",
    "jerk_mean", "jerk_std", "jerk_max",
    "curvature_mean", "curvature_std", "curvature_max",
    "path_length", "path_efficiency", "direction_changes",
    "centerline_dev_mean", "centerline_dev_std",
    "duration_ms",
    "tremor_power_8_12hz",
]

COMBINED_COLS = MOTOR_COLS + MECHANISM_COLS


# Attacks to evaluate. Order is preserved in the output JSON.
ATTACKS = [
    "visual",
    "replay",
    "replay_v1b",
    "replay_v1c",
    "replay_v2",
    "replay_timed",
    "warp_v1",
    "warp_t1",
]

# Allowlists whose session_ids must be excluded from the held-out human eval set.
# Union of these is the uniform exclusion set.
ALLOWLIST_FILES = [
    "experiments_replay/allowed_sessions_v1b.json",
    "experiments_replay/allowed_sessions_v1c.json",
    "experiments_replay/allowed_sessions_warp_t1.json",
]


def load_exclusion_set():
    """Return the union of all known source-bank session_ids."""
    ids = set()
    for rel in ALLOWLIST_FILES:
        path = os.path.join(REPO, rel)
        if not os.path.exists(path):
            print(f"  WARN: allowlist missing: {rel}")
            continue
        with open(path) as f:
            data = json.load(f)
            ids.update(data)
    return ids


def cv_auc(X, y, n_estimators=200, n_splits=5, seed=42):
    """Cross-validated AUC for an RF on (X, y). Returns (mean_auc, std_auc)."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=seed)),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc"], return_train_score=False)
    aucs = scores["test_roc_auc"]
    return float(aucs.mean()), float(aucs.std())


def eval_attack(df_humans, df_attack, attack_name):
    """Run three heads on (humans vs attack). Returns dict of head -> (auc, std, n_human, n_attack)."""
    sub = pd.concat([df_humans, df_attack], axis=0, ignore_index=True)
    sub["y"] = (sub["source"] == attack_name).astype(int)
    y = sub["y"].values
    n_human = int((y == 0).sum())
    n_attack = int((y == 1).sum())

    if n_attack < 5:
        return {
            "skipped": True,
            "reason": f"too few attack samples (n={n_attack})",
            "n_human": n_human,
            "n_attack": n_attack,
        }

    out = {"n_human": n_human, "n_attack": n_attack, "skipped": False, "heads": {}}
    for head_name, cols in [("motor", MOTOR_COLS), ("mechanism", MECHANISM_COLS), ("combined", COMBINED_COLS)]:
        X = sub[cols].fillna(0).values
        auc, std = cv_auc(X, y)
        out["heads"][head_name] = {"auc": auc, "std": std, "n_features": len(cols)}
    return out


def main():
    print("Building features from data/ ...")
    df = build_feature_dataframe(os.path.join(REPO, "data"))
    df = df[(df["completed"] == True) & (df["n_points"] >= 10)].copy()
    print(f"  Completed trajectories: {len(df)}")
    print(f"  Sources: {df['source'].value_counts().to_dict()}")

    exclusion = load_exclusion_set()
    print(f"\nExclusion set (union of attack source banks): {len(exclusion)} session_ids")

    df_humans = df[(df["source"] == "human") & (~df["session_id"].isin(exclusion))].copy()
    print(f"Held-out humans (disjoint from all banks): {len(df_humans)}")

    summary = {
        "exclusion_set_size": len(exclusion),
        "n_held_out_humans": len(df_humans),
        "motor_cols": MOTOR_COLS,
        "mechanism_cols": MECHANISM_COLS,
        "attacks": {},
    }

    for atk in ATTACKS:
        df_atk = df[df["source"] == atk]
        if len(df_atk) == 0:
            print(f"\n[{atk}] no data — skipping")
            summary["attacks"][atk] = {"skipped": True, "reason": "no data"}
            continue
        print(f"\n[{atk}] humans={len(df_humans)}  attack={len(df_atk)}")
        result = eval_attack(df_humans, df_atk, atk)
        summary["attacks"][atk] = result
        if not result.get("skipped"):
            for h in ("motor", "mechanism", "combined"):
                v = result["heads"][h]
                print(f"  {h:10s} AUC = {v['auc']:.4f}  ±{v['std']:.4f}")

    out_path = os.path.join(HERE, "two_head_eval.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
