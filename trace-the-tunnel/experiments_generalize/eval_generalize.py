"""
eval_generalize.py — humans-vs-generated classifier with GroupKFold(tunnel_id).

For each (split_id, model) with data in data/gen_{model}_{split_id}/:
  1. Load successful generated traces (from manifest) and human traces
     restricted to the split's test tunnels.
  2. Extract features (analysis/features.py — same as two_head_eval).
  3. Train three RFs (motor head, mechanism head, combined head).
  4. GroupKFold on tunnel_id so the classifier never sees the same
     tunnel in train + test; this is the "generalize to unseen tunnels"
     test condition.
  5. Emit:
       results/summary.csv — one row per (split, model) with AUCs +
                             per-stage failure counts from the manifest
       results/{split}_{model}_traces.csv — per-trace p_attack
       results/{split}_{model}_top.md — attacker wins / losses digest

Run modes:
  python eval_generalize.py                       # eval all (split, model) pairs found
  python eval_generalize.py --split split_a --model opus    # one cell
"""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import yaml

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "analysis"))
from features import extract_features  # noqa: E402

DATA_DIR = os.path.join(REPO, "data")
HUMAN_DIR = os.path.join(DATA_DIR, "human")
RESULTS_DIR = os.path.join(HERE, "results")

MECHANISM_COLS = ["dt_mean", "dt_std", "dt_min", "dt_max", "n_points"]
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


def load_features(json_paths, source_label):
    rows = []
    for p in json_paths:
        try:
            with open(p) as f:
                traj = json.load(f)
        except Exception:
            continue
        feat = extract_features(traj)
        if feat is None:
            continue
        feat["__path"] = p
        feat["__source"] = source_label
        rows.append(feat)
    return rows


def load_manifest(experiment_dir):
    """Return the latest manifest's parsed contents and a list of saved
    trace filenames classified as success."""
    manifests = sorted(glob.glob(os.path.join(experiment_dir, "manifest_*.json")))
    if not manifests:
        return None, []
    with open(manifests[-1]) as f:
        m = json.load(f)
    saved = [a["saved"] for a in m.get("attempts", []) if a.get("stage") == "success" and a.get("saved")]
    return m, saved


def cv_predict_with_auc(X, y, groups, seed=42, n_estimators=200, n_splits=5):
    """Cross-val predict_proba + per-fold AUC. Returns (mean_auc, std_auc, proba)."""
    pos_groups = len({groups[i] for i in range(len(y)) if y[i] == 1})
    neg_groups = len({groups[i] for i in range(len(y)) if y[i] == 0})
    eff_splits = max(2, min(n_splits, pos_groups, neg_groups))
    folds = list(GroupKFold(n_splits=eff_splits).split(X, y, groups))

    proba = np.full(len(y), np.nan)
    aucs = []
    for tr, te in folds:
        if len({y[i] for i in te}) < 2 or len({y[i] for i in tr}) < 2:
            continue
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=seed)),
        ])
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        proba[te] = p
        aucs.append(roc_auc_score(y[te], p))
    if not aucs:
        return float("nan"), float("nan"), proba
    return float(np.mean(aucs)), float(np.std(aucs)), proba


def eval_pair(split, model, demo_or_test="test"):
    """Run one (split, model) cell. demo_or_test selects which tunnel set
    the agent's traces should be associated with."""
    split_id = split["id"]
    target_tunnels = set(split[demo_or_test])
    exp_name = f"gen_{model}_{split_id}"
    exp_dir = os.path.join(DATA_DIR, exp_name)

    manifest, saved = load_manifest(exp_dir)
    if manifest is None:
        return {"status": "no_manifest", "experiment": exp_name}
    if not saved:
        return {
            "status": "no_successes",
            "experiment": exp_name,
            "counts": manifest.get("counts", {}),
            "n_attempts": manifest.get("n_attempts", 0),
        }

    gen_paths = [os.path.join(exp_dir, n) for n in saved if os.path.exists(os.path.join(exp_dir, n))]
    gen_rows = load_features(gen_paths, source_label="attack")
    gen_rows = [r for r in gen_rows if r.get("tunnel_id") in target_tunnels]

    human_paths = glob.glob(os.path.join(HUMAN_DIR, "*.json"))
    human_rows = load_features(human_paths, source_label="human")
    human_rows = [r for r in human_rows if r.get("tunnel_id") in target_tunnels]

    if not gen_rows or not human_rows:
        return {
            "status": "no_features",
            "experiment": exp_name,
            "n_gen": len(gen_rows),
            "n_human": len(human_rows),
        }

    df = pd.DataFrame(human_rows + gen_rows).fillna(0)
    df["y"] = (df["__source"] == "attack").astype(int)

    out = {
        "status": "ok",
        "experiment": exp_name,
        "split_id": split_id,
        "model": model,
        "tunnels_evaluated_on": sorted(target_tunnels),
        "n_human": len(human_rows),
        "n_attack": len(gen_rows),
        "counts": manifest.get("counts", {}),
        "n_attempts": manifest.get("n_attempts", 0),
    }
    if "n_attempts" in out and out["n_attempts"]:
        out["success_rate"] = round(out["counts"].get("success", 0) / out["n_attempts"], 3)

    groups = df["tunnel_id"].values
    y = df["y"].values
    head_protos = []
    for head, cols in [("motor", MOTOR_COLS), ("mechanism", MECHANISM_COLS), ("combined", COMBINED_COLS)]:
        X = df[cols].values
        mean, std, proba = cv_predict_with_auc(X, y, groups)
        out[f"{head}_auc"] = round(mean, 4) if not np.isnan(mean) else None
        out[f"{head}_auc_std"] = round(std, 4) if not np.isnan(std) else None
        head_protos.append((head, proba))

    # Per-trace CSV
    per_trace = pd.DataFrame({
        "trace_id": [os.path.basename(p) for p in df["__path"]],
        "source": df["__source"].values,
        "tunnel_id": df["tunnel_id"].values,
        "path": df["__path"].values,
    })
    for head, p in head_protos:
        per_trace[f"p_attack_{head}"] = p
    per_trace["correct@0.5"] = (
        ((per_trace["p_attack_combined"] >= 0.5) & (per_trace["source"] == "attack"))
        | ((per_trace["p_attack_combined"] < 0.5) & (per_trace["source"] == "human"))
    )

    tag = f"{split_id}_{model}" + ("_demo" if demo_or_test == "demo" else "")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    trace_path = os.path.join(RESULTS_DIR, f"{tag}_traces.csv")
    per_trace.to_csv(trace_path, index=False)
    out["traces_csv"] = trace_path

    # Top.md digest
    md_path = os.path.join(RESULTS_DIR, f"{tag}_top.md")
    write_top_md(md_path, per_trace, exp_name, out)
    out["top_md"] = md_path
    return out


def write_top_md(path, per_trace, exp_name, summary):
    pt = per_trace.dropna(subset=["p_attack_combined"]).copy()
    attack = pt[pt["source"] == "attack"].sort_values("p_attack_combined")
    human = pt[pt["source"] == "human"].sort_values("p_attack_combined", ascending=False)
    with open(path, "w") as f:
        f.write(f"# {exp_name} — top wins / losses\n\n")
        f.write(f"- combined AUC: {summary.get('combined_auc')} (motor {summary.get('motor_auc')}, mech {summary.get('mechanism_auc')})\n")
        f.write(f"- n_human: {summary['n_human']}  n_attack: {summary['n_attack']}\n")
        if "success_rate" in summary:
            f.write(f"- success_rate: {summary['success_rate']}  (counts: {summary['counts']})\n")
        f.write("\n## Top attacker wins (lowest p_attack on attack traces)\n")
        f.write("| tunnel_id | p_combined | path |\n|---|---|---|\n")
        for _, r in attack.head(10).iterrows():
            f.write(f"| {int(r['tunnel_id'])} | {r['p_attack_combined']:.3f} | `{r['path']}` |\n")
        f.write("\n## Top attacker losses (highest p_attack on attack traces)\n")
        f.write("| tunnel_id | p_combined | path |\n|---|---|---|\n")
        for _, r in attack.tail(10).iloc[::-1].iterrows():
            f.write(f"| {int(r['tunnel_id'])} | {r['p_attack_combined']:.3f} | `{r['path']}` |\n")
        f.write("\n## Top human false positives (humans most-confidently called attack)\n")
        f.write("| tunnel_id | p_combined | path |\n|---|---|---|\n")
        for _, r in human.head(10).iterrows():
            f.write(f"| {int(r['tunnel_id'])} | {r['p_attack_combined']:.3f} | `{r['path']}` |\n")


def write_summary(rows, path):
    """Flatten the eval_pair results into a summary.csv."""
    if not rows:
        return
    cols = [
        "experiment", "split_id", "model", "status",
        "n_attempts", "success_rate",
        "n_human", "n_attack",
        "motor_auc", "motor_auc_std",
        "mechanism_auc", "mechanism_auc_std",
        "combined_auc", "combined_auc_std",
    ]
    # also expand counts dict
    flat_rows = []
    for r in rows:
        flat = {c: r.get(c) for c in cols}
        counts = r.get("counts", {}) or {}
        for k in ("success", "solver_error", "malformed", "captcha_reject"):
            flat[f"n_{k}"] = counts.get(k, 0)
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=os.path.join(HERE, "splits.yaml"))
    ap.add_argument("--split", default=None, help="restrict to one split id")
    ap.add_argument("--model", default=None, help="restrict to one model")
    ap.add_argument("--demo", action="store_true",
                    help="evaluate against the demo tunnels (sanity check) instead of test")
    args = ap.parse_args()

    with open(args.splits) as f:
        cfg = yaml.safe_load(f)
    models = cfg["generation"]["models"]
    splits = cfg["splits"]

    sel_splits = [s for s in splits if args.split is None or s["id"] == args.split]
    sel_models = [m for m in models if args.model is None or m == args.model]

    rows = []
    for split in sel_splits:
        for model in sel_models:
            tag = "demo" if args.demo else "test"
            print(f"\n=== {split['id']} / {model} (eval on {tag} tunnels) ===")
            r = eval_pair(split, model, demo_or_test=("demo" if args.demo else "test"))
            print(f"  status: {r['status']}")
            if r["status"] == "ok":
                print(f"  AUC motor={r['motor_auc']} mech={r['mechanism_auc']} combined={r['combined_auc']}")
                print(f"  success_rate={r.get('success_rate')}  counts={r.get('counts')}")
            rows.append(r)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    write_summary(rows, summary_path)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
