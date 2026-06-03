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
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict, cross_validate
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
    "replay_cdp",
    "replay_hid",
    "replay_init",
    "rl_agent",
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


def cv_auc(X, y, n_estimators=200, n_splits=5, seed=42, groups=None):
    """Cross-validated AUC for an RF on (X, y). Returns (mean_auc, std_auc).

    If `groups` is provided, uses GroupKFold so samples with the same group
    label stay in the same fold. The fold count is clamped to the number of
    distinct groups within each class so a low-diversity attack (e.g.
    replay_v1c with 4 unique sources) doesn't produce all-NaN folds.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=seed)),
    ])
    if groups is not None:
        # GroupKFold needs n_splits ≤ n_unique_groups. Clamp by the smaller
        # of human-side and attack-side group counts, so each fold can have
        # at least one positive and one negative example.
        pos_groups = len(set(groups[i] for i in range(len(y)) if y[i] == 1))
        neg_groups = len(set(groups[i] for i in range(len(y)) if y[i] == 0))
        eff_splits = max(2, min(n_splits, pos_groups, neg_groups))
        splits = list(GroupKFold(n_splits=eff_splits).split(X, y, groups))
        # Drop folds where the test set has only one class (AUC undefined).
        valid = []
        for tr, te in splits:
            if len(set(y[te])) >= 2:
                valid.append((tr, te))
        if not valid:
            return float("nan"), float("nan")
        scores = cross_validate(model, X, y, cv=valid, scoring=["roc_auc"], return_train_score=False)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc"], return_train_score=False)
    aucs = scores["test_roc_auc"]
    return float(aucs.mean()), float(aucs.std())


# Attacks whose traces share source humans, so they need GroupKFold by
# source_session to avoid CV-leakage. Any replay-from-bank attack qualifies:
# multiple traces drawn from the same source human carry near-identical
# trajectory-shape features, so StratifiedKFold puts copies in different
# folds and the classifier memorizes the fingerprint instead of learning a
# real human-vs-attack discriminator.
#
# replay_init is the strongest case (byte-identical), but the same effect
# is present in lower-noise channels (cdp, hid) and in any solver that
# samples its source bank with replacement.
GROUP_BY_SOURCE_ATTACKS = {
    "replay_init", "replay_cdp", "replay_hid",
}


# Manifest-based source maps for solvers that recorded source_session
# directly in their results entries.
MANIFEST_SOURCE_MAP_ATTACKS = {"replay_cdp", "replay_hid", "replay_init"}


def _load_manifest_source_map(attack):
    out = {}
    d = os.path.join(REPO, "data", attack)
    if not os.path.isdir(d):
        return out
    for mf in sorted(os.listdir(d)):
        if not (mf.startswith("manifest_") and mf.endswith(".json")):
            continue
        try:
            j = json.load(open(os.path.join(d, mf)))
        except Exception:
            continue
        for r in j.get("results", []):
            if r.get("saved") and r.get("source_session"):
                out[r["saved"].replace(".json", "")] = r["source_session"]
    return out


def _load_reconstructed_source_map(attack, allowed_sessions_path=None):
    """For attacks without manifest source mapping, reconstruct it by
    matching each replay's mousemove (x, y) sequence to the closest bank
    human's. Channel noise is small (< 2 px / < 0.5 ms per event), so the
    correct source always wins by orders of magnitude in L2 distance.

    Bank candidates per tunnel are restricted to `allowed_sessions_path` if
    given; otherwise any completed human for that tunnel is a candidate.
    """
    import numpy as np
    out = {}
    attack_dir = os.path.join(REPO, "data", attack)
    human_dir = os.path.join(REPO, "data", "human")
    if not os.path.isdir(attack_dir) or not os.path.isdir(human_dir):
        return out

    allowed = None
    if allowed_sessions_path:
        path = os.path.join(REPO, allowed_sessions_path)
        if os.path.exists(path):
            allowed = set(json.load(open(path)))

    # Index human bank by tunnel_id
    bank_by_tid = {}
    for fname in os.listdir(human_dir):
        if not fname.endswith(".json"):
            continue
        try:
            d = json.load(open(os.path.join(human_dir, fname)))
        except Exception:
            continue
        if not d.get("completed"):
            continue
        sid = d.get("session_id")
        if allowed is not None and sid not in allowed:
            continue
        tid = d.get("tunnel_id")
        moves = [(e["x"], e["y"]) for e in d.get("events", []) if e["event_type"] == "mousemove"]
        if not moves:
            continue
        bank_by_tid.setdefault(tid, []).append((sid, np.array(moves, dtype=float)))

    # Match each replay to its closest bank entry
    for fname in os.listdir(attack_dir):
        if not fname.endswith(".json") or fname.startswith("manifest_"):
            continue
        try:
            d = json.load(open(os.path.join(attack_dir, fname)))
        except Exception:
            continue
        if not d.get("completed"):
            continue
        rid = fname.replace(".json", "")
        tid = d.get("tunnel_id")
        candidates = bank_by_tid.get(tid, [])
        if not candidates:
            continue
        my_moves = np.array(
            [(e["x"], e["y"]) for e in d.get("events", []) if e["event_type"] == "mousemove"],
            dtype=float,
        )
        if len(my_moves) == 0:
            continue
        best_sid, best_score = None, float("inf")
        for sid, ref_moves in candidates:
            n = min(len(my_moves), len(ref_moves))
            if n == 0:
                continue
            diff = my_moves[:n] - ref_moves[:n]
            score = float(np.mean(diff * diff))
            if score < best_score:
                best_score, best_sid = score, sid
        if best_sid is not None:
            out[rid] = best_sid
    return out


# allowed_sessions file per replay attack, for narrowing the reconstruction
# candidate pool. Currently empty: all kept replay attacks (cdp/hid/init)
# carry source_session in their manifests, so reconstruction is unused.
RECONSTRUCT_ALLOWLISTS = {}


def load_attack_source_map(attack):
    """Best-effort: returns a dict mapping replay session_id → source human
    session_id. Falls back to reconstruction by event-matching for older
    attacks without recorded source_session fields."""
    if attack in MANIFEST_SOURCE_MAP_ATTACKS:
        return _load_manifest_source_map(attack)
    if attack in RECONSTRUCT_ALLOWLISTS:
        return _load_reconstructed_source_map(attack, RECONSTRUCT_ALLOWLISTS[attack])
    return {}


def eval_attack(df_humans, df_attack, attack_name, source_map=None):
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

    # Compute groups for GROUP_BY_SOURCE_ATTACKS. Group = source human's
    # session_id for replay rows; the human's own session_id for human rows.
    # If source_map is empty/None (manifest missing AND reconstruction
    # failed), fall back to StratifiedKFold and flag in the output.
    groups = None
    cv_strategy = "stratified_kfold"
    n_unique_sources = None
    if attack_name in GROUP_BY_SOURCE_ATTACKS:
        if source_map:
            gs = []
            for _, row in sub.iterrows():
                if row["source"] == attack_name:
                    gs.append(source_map.get(row["session_id"], row["session_id"]))
                else:
                    gs.append(row["session_id"])
            groups = np.array(gs)
            cv_strategy = "group_kfold_by_source"
            # Count distinct attack-side groups so we can sanity-check fold counts.
            atk_groups = [groups[i] for i in range(len(sub)) if sub.iloc[i]["source"] == attack_name]
            n_unique_sources = len(set(atk_groups))
        else:
            cv_strategy = "stratified_kfold_NO_SOURCE_MAP"

    out = {
        "n_human": n_human,
        "n_attack": n_attack,
        "skipped": False,
        "cv": cv_strategy,
        "n_unique_sources": n_unique_sources,
        "heads": {},
    }
    for head_name, cols in [("motor", MOTOR_COLS), ("mechanism", MECHANISM_COLS), ("combined", COMBINED_COLS)]:
        X = sub[cols].fillna(0).values
        auc, std = cv_auc(X, y, groups=groups)
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

    # Pre-load source mappings per attack (used for GroupKFold)
    source_maps = {}
    for atk in ATTACKS:
        if atk in GROUP_BY_SOURCE_ATTACKS:
            m = load_attack_source_map(atk)
            if m:
                source_maps[atk] = m
                print(f"  [{atk}] source map: {len(m)} replays → {len(set(m.values()))} unique sources")

    for atk in ATTACKS:
        df_atk = df[df["source"] == atk]
        if len(df_atk) == 0:
            print(f"\n[{atk}] no data — skipping")
            summary["attacks"][atk] = {"skipped": True, "reason": "no data"}
            continue
        print(f"\n[{atk}] humans={len(df_humans)}  attack={len(df_atk)}")
        result = eval_attack(df_humans, df_atk, atk, source_map=source_maps.get(atk))
        summary["attacks"][atk] = result
        if not result.get("skipped"):
            cv_tag = f" [{result.get('cv','?')}]"
            for h in ("motor", "mechanism", "combined"):
                v = result["heads"][h]
                print(f"  {h:10s} AUC = {v['auc']:.4f}  ±{v['std']:.4f}{cv_tag if h=='motor' else ''}")

    out_path = os.path.join(HERE, "two_head_eval.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
