"""
Generate PPO policy trajectories in simulation, plot against human data,
and run the classifier to measure human-likeness.

Usage:
  cd trace-the-tunnel
  python experiments_rl/eval_ppo.py [--policy ppo_policy_best.pt] [--n_trials 10]
"""

import argparse
import json
import math
import os
import random
import sys
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

from policy import TunnelPolicy, load_policy
from tunnel_env import (
    TunnelEnv, generate_tunnel, is_inside_tunnel,
    TUNNEL_WIDTH, CANVAS_W, CANVAS_H,
)
from features import extract_features, events_to_arrays

warnings.filterwarnings("ignore")

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "tunnel_config.json")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "human")
OUT_PATH    = os.path.join(os.path.dirname(__file__), "best_img.png")

DT_MEAN_MS = 8.26
DT_STD_MS  = 0.92
DT_MIN_MS  = 5.5
DT_MAX_MS  = 20.0


def sample_dt(rng):
    dt = rng.gauss(DT_MEAN_MS, DT_STD_MS)
    if rng.random() < 0.03:
        dt += rng.expovariate(1 / 15.0)
    return max(DT_MIN_MS, min(DT_MAX_MS, dt))


def generate_trajectory(env, policy, rng, jitter_px=0.4):
    """Run policy in simulation; return (events_list, success)."""
    obs, _ = env.reset()
    events = [{"x": env._x, "y": env._y, "dt_ms": 0.0, "event_type": "mousedown"}]
    t_cum = 0.0
    done = truncated = False

    while not (done or truncated):
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = policy.get_action(obs_t, deterministic=False)
        obs, _, done, truncated, info = env.step(action.squeeze(0).numpy())

        jx = rng.gauss(0.0, jitter_px)
        jy = rng.gauss(0.0, jitter_px)
        xe = float(np.clip(env._x + jx, 0, CANVAS_W))
        ye = float(np.clip(env._y + jy, 0, CANVAS_H))
        if not is_inside_tunnel(xe, ye, env.centerline):
            xe, ye = env._x, env._y

        dt = sample_dt(rng)
        t_cum += dt
        events.append({"x": xe, "y": ye, "dt_ms": dt,
                        "timestamp": t_cum, "event_type": "mousemove"})

    success = done and info.get("reason") == "success"
    return events, success


def events_to_traj_dict(events, tid, control_points):
    """Wrap events into the dict format expected by extract_features."""
    # Convert dt_ms chain to absolute timestamps if not already set
    t = 0.0
    for e in events:
        if "timestamp" not in e:
            t += e.get("dt_ms", 0.0)
            e["timestamp"] = t
    return {
        "tunnel_id": tid,
        "source": "rl_agent",
        "completed": True,
        "control_points": control_points,
        "events": events,
    }


# ---------------------------------------------------------------------------
# Load human data grouped by tunnel
# ---------------------------------------------------------------------------

def load_human_by_tunnel():
    by_tunnel = {}
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(DATA_DIR, fname)) as f:
            traj = json.load(f)
        if not traj.get("completed"):
            continue
        tid = traj.get("tunnel_id")
        if tid is None:
            continue
        by_tunnel.setdefault(tid, []).append(traj)
    return by_tunnel


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def draw_tunnel(ax, control_points, tunnel_width):
    n_segs = len(control_points) // 4
    cx, cy = [], []
    for s in range(n_segs):
        p0, p1, p2, p3 = (control_points[s*4], control_points[s*4+1],
                           control_points[s*4+2], control_points[s*4+3])
        for i in range(100):
            t = i / 99
            u = 1 - t
            cx.append(u**3*p0["x"] + 3*u**2*t*p1["x"] + 3*u*t**2*p2["x"] + t**3*p3["x"])
            cy.append(u**3*p0["y"] + 3*u**2*t*p1["y"] + 3*u*t**2*p2["y"] + t**3*p3["y"])
    ax.fill_between(cx, np.array(cy) - tunnel_width, np.array(cy) + tunnel_width,
                    color="#cccccc", alpha=0.35, zorder=0)
    ax.plot(cx, cy, color="gray", linewidth=1, linestyle="--", alpha=0.5, zorder=1)
    return cx, cy


def make_plot(human_by_tunnel, agent_by_tunnel, tunnel_ids, out_path):
    n = len(tunnel_ids)
    fig, axes = plt.subplots(n, 1, figsize=(8, 4.5 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle("Human (blue) vs RL Agent (red) trajectories", fontsize=13, y=1.002)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#44bb44",
               markersize=9, label="start"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ee4444",
               markersize=9, label="end"),
        Line2D([0], [0], color="#4fc3f7", linewidth=1.5, alpha=0.7, label="human"),
        Line2D([0], [0], color="#ff5555", linewidth=1.5, label="RL agent"),
    ]

    for ax, tid in zip(axes, tunnel_ids):
        ax.set_facecolor("white")
        ax.set_title(f"Tunnel {tid}", fontsize=11)
        ax.set_xlim(0, CANVAS_W)
        ax.set_ylim(CANVAS_H, 0)  # flip y to match canvas coords
        ax.axis("off")

        # Draw tunnel shape from first human traj with control points
        cp = None
        tw = TUNNEL_WIDTH
        for traj in human_by_tunnel.get(tid, []):
            if traj.get("control_points"):
                cp = traj["control_points"]
                tw = traj.get("tunnel_width", TUNNEL_WIDTH)
                break
        if cp:
            draw_tunnel(ax, cp, tw)

        # Human trajectories
        for traj in human_by_tunnel.get(tid, []):
            result = events_to_arrays(traj["events"])
            if result is None:
                continue
            x, y, _ = result
            ax.plot(x, y, color="#4fc3f7", alpha=0.45, linewidth=1.0, zorder=2)

        # Agent trajectories
        for traj in agent_by_tunnel.get(tid, []):
            result = events_to_arrays(traj["events"])
            if result is None:
                continue
            x, y, _ = result
            ax.plot(x, y, color="#ff5555", alpha=0.85, linewidth=1.4, zorder=3)

        # Start / end dots (from first human traj)
        ref = human_by_tunnel.get(tid, [None])[0]
        if ref:
            res = events_to_arrays(ref["events"])
            if res is not None:
                hx, hy, _ = res
                ax.scatter([hx[0]], [hy[0]], color="#44bb44", s=80, zorder=5)
                ax.scatter([hx[-1]], [hy[-1]], color="#ee4444", s=80, zorder=5)

        ax.legend(handles=legend_elements, loc="upper right", fontsize=7,
                  framealpha=0.8, borderpad=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def run_classifier(human_features, agent_features):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

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

    for f in human_features:
        f["is_agent"] = 0
    for f in agent_features:
        f["is_agent"] = 1

    df = pd.DataFrame(human_features + agent_features)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0).values
    y = df["is_agent"].values

    n_splits = min(5, int(np.bincount(y).min()))
    if n_splits < 2:
        print("Not enough samples for cross-validation.")
        return

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    classifiers = {
        "Logistic Regression": Pipeline([("s", StandardScaler()),
                                         ("c", LogisticRegression(max_iter=1000, random_state=42))]),
        "Random Forest":       Pipeline([("s", StandardScaler()),
                                         ("c", RandomForestClassifier(n_estimators=100, random_state=42))]),
        "Gradient Boosting":   Pipeline([("s", StandardScaler()),
                                         ("c", GradientBoostingClassifier(n_estimators=100, random_state=42))]),
    }

    print("\n" + "=" * 58)
    print("CLASSIFIER RESULTS  (human=0, RL agent=1)")
    print(f"  {len(human_features)} human traces  |  {len(agent_features)} agent traces")
    print("=" * 58)

    for name, pipe in classifiers.items():
        y_pred = cross_val_predict(pipe, X, y, cv=cv)
        scores = cross_validate(pipe, X, y, cv=cv,
                                scoring=["accuracy", "f1", "roc_auc"])
        try:
            y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, y_prob)
        except Exception:
            auc = scores["test_roc_auc"].mean()

        print(f"\n--- {name} ---")
        print(f"Accuracy : {accuracy_score(y, y_pred):.3f}")
        print(f"ROC AUC  : {auc:.3f}")
        print(f"CV Acc   : {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}")
        print(classification_report(y, y_pred, target_names=["human", "agent"]))
        print("Confusion matrix:")
        print(confusion_matrix(y, y_pred))

    # Feature importance from Random Forest (trained on full dataset)
    print("\n" + "=" * 58)
    print("FEATURE IMPORTANCE  (Random Forest, trained on full dataset)")
    print("=" * 58)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_
    ranked = sorted(zip(available, importances), key=lambda x: x[1], reverse=True)

    # Also compute per-feature mean difference (agent - human) for direction
    df_h = df[df["is_agent"] == 0][available].mean()
    df_a = df[df["is_agent"] == 1][available].mean()

    print(f"\n{'Rank':<5} {'Feature':<26} {'Importance':>10}  {'Human mean':>11}  {'Agent mean':>11}  {'Diff':>8}")
    print("-" * 78)
    for rank, (feat, imp) in enumerate(ranked, 1):
        h_val = df_h[feat]
        a_val = df_a[feat]
        diff = a_val - h_val
        print(f"{rank:<5} {feat:<26} {imp:>10.4f}  {h_val:>11.4f}  {a_val:>11.4f}  {diff:>+8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",   default="ppo_policy_best.pt")
    parser.add_argument("--n_trials", type=int, default=10,
                        help="Agent rollouts per tunnel")
    parser.add_argument("--tunnels",  type=int, nargs="+",
                        default=[0, 2, 4, 7, 8])
    args = parser.parse_args()

    policy_path = os.path.join(MODEL_DIR, args.policy)
    if not os.path.exists(policy_path):
        print(f"Policy not found: {policy_path}")
        sys.exit(1)

    policy = load_policy(policy_path)
    policy.eval()
    print(f"Loaded: {policy_path}")

    with open(CONFIG_PATH) as f:
        config = json.load(f)
    seed_map = {t["tunnel_id"]: t["seed"] for t in config["tunnels"]}

    human_by_tunnel = load_human_by_tunnel()
    agent_by_tunnel = {}

    print(f"\nGenerating {args.n_trials} trajectories per tunnel ...")
    rng = random.Random(42)

    for tid in args.tunnels:
        seed = seed_map.get(tid, tid)
        successes, attempts = [], 0
        while len(successes) < args.n_trials and attempts < args.n_trials * 5:
            attempts += 1
            env = TunnelEnv(tunnel_id=tid, seed=seed)
            events, ok = generate_trajectory(env, policy, rng)
            if ok:
                # Get control_points from env or human data
                cp = None
                for traj in human_by_tunnel.get(tid, []):
                    if traj.get("control_points"):
                        cp = traj["control_points"]
                        break
                successes.append(events_to_traj_dict(events, tid, cp or []))
        agent_by_tunnel[tid] = successes
        print(f"  Tunnel {tid}: {len(successes)}/{args.n_trials} succeeded "
              f"({attempts} attempts)")

    # Plot
    make_plot(human_by_tunnel, agent_by_tunnel, args.tunnels, OUT_PATH)

    # Classifier
    human_feats, agent_feats = [], []
    for tid in args.tunnels:
        for traj in human_by_tunnel.get(tid, []):
            f = extract_features(traj)
            if f:
                human_feats.append(f)
        for traj in agent_by_tunnel.get(tid, []):
            f = extract_features(traj)
            if f:
                agent_feats.append(f)

    run_classifier(human_feats, agent_feats)


if __name__ == "__main__":
    main()
