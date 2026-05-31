"""
Fair evaluation for replay_init.

The standard analysis (analysis/two_head_eval.py) uses StratifiedKFold
over individual trace rows. That's the right thing for every other
attack we've measured, because each replay's features carry channel-
specific noise — the 90 replay_cdp traces aren't byte-identical to each
other; each has its own dispatcher fingerprint.

replay_init is the one attack where the recorded features are
byte-identical to the source human's. With 8 replays per tunnel and
only 3 unique humans per tunnel in the bank, each unique human's
feature vector appears in the dataset ~3 times (with identical values).
StratifiedKFold splits these copies across folds; the classifier
memorizes the fingerprint in training and reads it back in test.
Result: AUC 0.97, which says nothing about whether the replay is
detectable — it says the classifier can solve a copy-memorization task.

The fair eval here:
  - GroupKFold by source_session, so all replays of one human stay in
    the same fold. The classifier cannot use train-set copies as a key
    to identify test-set copies.
  - Also reports a deduplicated version (one replay per unique source),
    so the apparent class size matches the underlying number of
    distinct fingerprints.

Run:
  python experiments_replay_init/fair_eval.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "analysis"))
from features import build_feature_dataframe  # noqa: E402
from two_head_eval import COMBINED_COLS, MECHANISM_COLS, MOTOR_COLS, load_exclusion_set  # noqa: E402


def load_source_map():
    """Map saved replay_init session_id → source_session_id via manifests."""
    out = {}
    out_dir = os.path.join(REPO, "data", "replay_init")
    for mf in sorted(os.listdir(out_dir)):
        if not mf.startswith("manifest_") or not mf.endswith(".json"):
            continue
        d = json.load(open(os.path.join(out_dir, mf)))
        for r in d.get("results", []):
            if r.get("saved"):
                out[r["saved"].replace(".json", "")] = r["source_session"]
    return out


def main():
    df = build_feature_dataframe(os.path.join(REPO, "data"))
    df = df[(df["completed"] == True) & (df["n_points"] >= 10)].copy()
    excl = load_exclusion_set()
    humans = df[(df["source"] == "human") & (~df["session_id"].isin(excl))]
    init = df[df["source"] == "replay_init"]
    src_map = load_source_map()
    init_src = init["session_id"].map(src_map)
    init_with_src = init.assign(__source_session=init_src.values)

    print(f"held-out humans: {len(humans)}")
    print(f"replay_init traces: {len(init)} ({init_with_src['__source_session'].nunique()} unique sources)")
    print()

    # ---- 1. Standard pipeline (current two_head_eval) ----
    print("=== Standard StratifiedKFold (matches analysis/two_head_eval.py) ===")
    sub = pd.concat([humans, init], ignore_index=True)
    sub["y"] = (sub["source"] == "replay_init").astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, cols in [("motor", MOTOR_COLS), ("mechanism", MECHANISM_COLS), ("combined", COMBINED_COLS)]:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        aucs = cross_val_score(clf, sub[cols].fillna(0).values, sub["y"].values, cv=cv, scoring="roc_auc")
        print(f"  {name:10s}  AUC = {aucs.mean():.4f}  ±{aucs.std():.4f}")

    # ---- 2. GroupKFold by source_session — the honest eval ----
    print()
    print("=== GroupKFold by source_session (replays of one human stay in same fold) ===")
    sub2 = pd.concat([humans.assign(__source_session=humans["session_id"]), init_with_src], ignore_index=True)
    sub2["y"] = (sub2["source"] == "replay_init").astype(int)
    groups = sub2["__source_session"].values
    splits = list(GroupKFold(n_splits=5).split(np.zeros(len(sub2)), sub2["y"].values, groups))
    for name, cols in [("motor", MOTOR_COLS), ("mechanism", MECHANISM_COLS), ("combined", COMBINED_COLS)]:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        aucs = cross_val_score(clf, sub2[cols].fillna(0).values, sub2["y"].values, cv=splits, scoring="roc_auc")
        print(f"  {name:10s}  AUC = {aucs.mean():.4f}  ±{aucs.std():.4f}")

    # ---- 3. Deduplicated — one trace per unique source ----
    print()
    print("=== Deduped to one replay per unique source ===")
    init_dedup = init_with_src.drop_duplicates(subset="__source_session")
    sub3 = pd.concat([humans, init_dedup.drop(columns="__source_session")], ignore_index=True)
    sub3["y"] = (sub3["source"] == "replay_init").astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, cols in [("motor", MOTOR_COLS), ("mechanism", MECHANISM_COLS), ("combined", COMBINED_COLS)]:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        aucs = cross_val_score(clf, sub3[cols].fillna(0).values, sub3["y"].values, cv=cv, scoring="roc_auc")
        print(f"  {name:10s}  AUC = {aucs.mean():.4f}  ±{aucs.std():.4f}")

    # ---- 4. Sanity check: max abs feature diff between replay and its source ----
    print()
    print("=== Per-feature parity check (replay row vs its source human) ===")
    bank = df[(df["source"] == "human") & (df["session_id"].isin(excl))]
    diffs = {c: [] for c in COMBINED_COLS}
    n = 0
    for _, r in init_with_src.iterrows():
        src_row = bank[bank["session_id"] == r["__source_session"]]
        if len(src_row) != 1:
            continue
        n += 1
        for c in COMBINED_COLS:
            diffs[c].append(abs(float(r[c]) - float(src_row.iloc[0][c])))
    if n:
        print(f"  paired {n} replays with their source")
        max_per_feat = sorted(((c, max(v)) for c, v in diffs.items()), key=lambda kv: -kv[1])[:5]
        for c, v in max_per_feat:
            print(f"  max |Δ{c}| over all pairs = {v:.6e}")


if __name__ == "__main__":
    main()
