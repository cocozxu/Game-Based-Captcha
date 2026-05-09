"""
Visualization of trajectory data and feature distributions.

Generates plots comparing human vs agent trajectories.
Run: python analysis/visualize.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams["figure.dpi"] = 120
matplotlib.rcParams["figure.figsize"] = (12, 8)

from features import build_feature_dataframe, load_trajectories, events_to_arrays


BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Trajectory overlay plot — same tunnel, human vs agent
# ---------------------------------------------------------------------------

def plot_trajectory_overlays():
    """Plot raw trajectories overlaid on the tunnel for each tunnel_id."""
    sources = {}
    for source_dir in sorted(os.listdir(BASE_DIR)):
        full = os.path.join(BASE_DIR, source_dir)
        if not os.path.isdir(full) or source_dir == "logs":
            continue
        sources[source_dir] = load_trajectories(full)

    # Group by tunnel_id
    by_tunnel = {}
    for src, trajs in sources.items():
        for traj in trajs:
            if not traj.get("completed"):
                continue
            tid = traj.get("tunnel_id")
            if tid is None:
                continue
            by_tunnel.setdefault(tid, []).append((src, traj))

    for tid in sorted(by_tunnel.keys()):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor("#16213e")
        ax.set_xlim(0, 600)
        ax.set_ylim(350, 0)  # flip y
        ax.set_title(f"Tunnel {tid} — Trajectory Overlay")
        ax.set_xlabel("x (canvas)")
        ax.set_ylabel("y (canvas)")

        # Draw centerline from first trajectory's control points
        first_traj = by_tunnel[tid][0][1]
        cp = first_traj.get("control_points", [])
        if len(cp) >= 4:
            n_segs = len(cp) // 4
            cx, cy = [], []
            for s in range(n_segs):
                p0, p1, p2, p3 = cp[s*4], cp[s*4+1], cp[s*4+2], cp[s*4+3]
                for i in range(100):
                    t = i / 99
                    u = 1 - t
                    cx.append(u**3*p0["x"] + 3*u**2*t*p1["x"] + 3*u*t**2*p2["x"] + t**3*p3["x"])
                    cy.append(u**3*p0["y"] + 3*u**2*t*p1["y"] + 3*u*t**2*p2["y"] + t**3*p3["y"])
            tw = first_traj.get("tunnel_width", 38)
            ax.plot(cx, cy, color="gray", linewidth=tw*1.5, alpha=0.15, solid_capstyle="round")
            ax.plot(cx, cy, color="gray", linewidth=1, alpha=0.4, linestyle="--")

        colors = {"human": "#4fc3f7", "agent": "#ff7043", "agent_humanlike": "#ab47bc"}
        for src, traj in by_tunnel[tid]:
            result = events_to_arrays(traj["events"])
            if result is None:
                continue
            x, y, t_arr = result
            c = colors.get(src, "#888888")
            ax.plot(x, y, color=c, alpha=0.5, linewidth=1.2, label=src)

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"overlay_tunnel_{tid}.png"))
        plt.close(fig)
        print(f"  Saved overlay_tunnel_{tid}.png")


# ---------------------------------------------------------------------------
# 2. Feature distribution comparisons
# ---------------------------------------------------------------------------

def plot_feature_distributions(df):
    """Boxplots and histograms comparing features across sources."""
    feature_cols = [
        "speed_mean", "speed_std", "accel_std", "jerk_std",
        "dt_mean", "dt_std", "path_efficiency", "curvature_mean",
        "centerline_dev_mean", "centerline_dev_std",
        "direction_changes", "duration_ms", "tremor_power_8_12hz",
    ]
    existing = [c for c in feature_cols if c in df.columns]

    sources = df["source"].unique()
    n_feat = len(existing)
    n_cols = 3
    n_rows = (n_feat + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    axes = axes.flatten()

    colors = {"human": "#4fc3f7", "agent": "#ff7043", "agent_humanlike": "#ab47bc"}

    for i, feat in enumerate(existing):
        ax = axes[i]
        for src in sorted(sources):
            vals = df[df["source"] == src][feat].dropna()
            c = colors.get(src, "#888888")
            ax.hist(vals, bins=15, alpha=0.5, label=src, color=c, density=True)
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions: Human vs Agent", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "feature_distributions.png"))
    plt.close(fig)
    print("  Saved feature_distributions.png")


# ---------------------------------------------------------------------------
# 3. Speed profile comparison
# ---------------------------------------------------------------------------

def plot_speed_profiles():
    """Plot speed over normalized time for a few trajectories per source."""
    sources_data = {}
    for source_dir in sorted(os.listdir(BASE_DIR)):
        full = os.path.join(BASE_DIR, source_dir)
        if not os.path.isdir(full) or source_dir == "logs":
            continue
        trajs = [t for t in load_trajectories(full) if t.get("completed")]
        sources_data[source_dir] = trajs

    fig, axes = plt.subplots(1, len(sources_data), figsize=(6 * len(sources_data), 5), squeeze=False)
    colors = {"human": "#4fc3f7", "agent": "#ff7043", "agent_humanlike": "#ab47bc"}

    for idx, (src, trajs) in enumerate(sorted(sources_data.items())):
        ax = axes[0][idx]
        ax.set_title(f"{src} speed profiles")
        ax.set_xlabel("Normalized time")
        ax.set_ylabel("Speed (px/ms)")
        c = colors.get(src, "#888")

        for traj in trajs[:10]:  # plot up to 10
            result = events_to_arrays(traj["events"])
            if result is None:
                continue
            x, y, t_arr = result
            dt = np.diff(t_arr)
            dt = np.where(dt == 0, 1e-3, dt)
            speed = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt
            t_norm = np.linspace(0, 1, len(speed))
            ax.plot(t_norm, speed, color=c, alpha=0.3, linewidth=1)

    fig.suptitle("Speed Profiles Over Normalized Time", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "speed_profiles.png"))
    plt.close(fig)
    print("  Saved speed_profiles.png")


# ---------------------------------------------------------------------------
# 4. Inter-event timing distribution
# ---------------------------------------------------------------------------

def plot_timing_distributions():
    """Histogram of inter-event intervals (dt) per source."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"human": "#4fc3f7", "agent": "#ff7043", "agent_humanlike": "#ab47bc"}

    for source_dir in sorted(os.listdir(BASE_DIR)):
        full = os.path.join(BASE_DIR, source_dir)
        if not os.path.isdir(full) or source_dir == "logs":
            continue
        trajs = [t for t in load_trajectories(full) if t.get("completed")]
        all_dt = []
        for traj in trajs:
            result = events_to_arrays(traj["events"])
            if result is None:
                continue
            _, _, t_arr = result
            dt = np.diff(t_arr)
            all_dt.extend(dt.tolist())

        if all_dt:
            c = colors.get(source_dir, "#888")
            ax.hist(all_dt, bins=100, alpha=0.5, label=source_dir, color=c,
                    density=True, range=(0, min(100, np.percentile(all_dt, 99))))

    ax.set_title("Inter-Event Interval Distribution")
    ax.set_xlabel("dt (ms)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "timing_distribution.png"))
    plt.close(fig)
    print("  Saved timing_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building features...")
    df = build_feature_dataframe(BASE_DIR)
    print(f"  {len(df)} trajectories loaded")

    print("\nGenerating plots...")
    plot_trajectory_overlays()
    plot_feature_distributions(df)
    plot_speed_profiles()
    plot_timing_distributions()

    print(f"\nAll plots saved to {OUT_DIR}/")
