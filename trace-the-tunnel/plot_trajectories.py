"""
Trajectory distribution analysis: human vs agent.

Figure 1 — Feature distributions: KDE overlays of per-session features
           (speed, timing, tremor, path length, centerline deviation)
           split by source type.

Figure 2 — Speed profile along arc: median speed at each arc-progress
           bin, showing how each group accelerates/decelerates through
           the tunnel.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).parent / "analysis"))
from features import build_feature_dataframe, load_trajectories, events_to_arrays

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR  = Path(__file__).parent

AGENT_DIRS = ["visual", "replay_timed", "replay", "visual_haiku", "visual_sonnet"]

# Colors per source — human always blue, agents in warm tones
SOURCE_COLORS = {
    "human":          "#2166ac",
    "visual":         "#d6604d",
    "replay_timed":   "#f4a582",
    "replay":         "#e08214",
    "visual_haiku":   "#8e0152",
    "visual_sonnet":  "#c51b7d",
    None:             "#2166ac",   # human sessions with null source
}
SOURCE_LABELS = {
    "human":         "Human",
    "visual":        "Visual (Sonnet)",
    "replay_timed":  "Replay (timed)",
    "replay":        "Replay",
    "visual_haiku":  "Visual (Haiku)",
    "visual_sonnet": "Visual (Sonnet no-ex)",
    None:            "Human",
}

SAMPLES_PER_SEG = 80


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_df():
    df = build_feature_dataframe(str(DATA_DIR))
    df = df[df["completed"] == True].copy()

    # Normalise source: human sessions sometimes have source=None
    df["source"] = df["source"].apply(
        lambda s: "human" if (s is None or str(s).lower() == "human") else s
    )

    # Keep only sources we care about
    keep = {"human"} | set(AGENT_DIRS)
    df = df[df["source"].isin(keep)].copy()

    print("Sessions per source:")
    print(df["source"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# Figure 1 — Feature KDE distributions
# ---------------------------------------------------------------------------

FEATURES = [
    ("speed_mean",           "Speed mean (px/ms)",         None),
    ("dt_mean",              "Inter-event dt mean (ms)",   None),
    ("path_length",          "Path length (px)",           None),
    ("centerline_dev_mean",  "Centerline deviation mean (px)", None),
    ("tremor_power_8_12hz",  "Tremor power 8–12 Hz",       None),
    ("duration_ms",          "Duration (ms)",              None),
]


def _kde_curve(values, lo=None, hi=None, n=300):
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return None, None
    lo = lo if lo is not None else values.min()
    hi = hi if hi is not None else values.max()
    if lo >= hi:
        return None, None
    xs = np.linspace(lo, hi, n)
    try:
        kde = gaussian_kde(values, bw_method="scott")
        return xs, kde(xs)
    except Exception:
        return None, None


def plot_feature_distributions(df):
    sources_ordered = ["human"] + [s for s in AGENT_DIRS if s in df["source"].unique()]
    n_feat = len(FEATURES)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle("Feature Distributions: Human vs Agent (per-session KDE)", fontsize=13)
    axes = axes.flatten()

    for ax, (col, label, _) in zip(axes, FEATURES):
        # Determine shared x range across all sources
        all_vals = df[col].dropna()
        # Clip extreme outliers for display (99th percentile)
        hi_clip = np.percentile(all_vals, 99)
        lo_clip = max(0, all_vals.min())

        for src in sources_ordered:
            vals = df[df["source"] == src][col].dropna().values
            vals = vals[vals <= hi_clip]
            if len(vals) < 3:
                continue
            xs, ys = _kde_curve(vals, lo=lo_clip, hi=hi_clip)
            if xs is None:
                continue
            color = SOURCE_COLORS.get(src, "#888")
            lw = 2.5 if src == "human" else 1.6
            ls = "-"  if src == "human" else "--"
            ax.plot(xs, ys, color=color, lw=lw, ls=ls,
                    label=SOURCE_LABELS.get(src, src))
            ax.fill_between(xs, ys, alpha=0.08, color=color)

        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_xlim(lo_clip, hi_clip)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=8, framealpha=0.7)
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Speed profile along arc
# ---------------------------------------------------------------------------

def _centerline_from_cps(cps):
    def bez(p0, p1, p2, p3, t):
        u = 1 - t
        return (u**3*p0[0] + 3*u**2*t*p1[0] + 3*u*t**2*p2[0] + t**3*p3[0],
                u**3*p0[1] + 3*u**2*t*p1[1] + 3*u*t**2*p2[1] + t**3*p3[1])
    pts = [(c["x"], c["y"]) for c in cps]
    cl = []
    for s in range(len(pts) // 4):
        p0, p1, p2, p3 = pts[s*4], pts[s*4+1], pts[s*4+2], pts[s*4+3]
        for i in range(SAMPLES_PER_SEG + 1):
            if cl and i == 0:
                continue
            cl.append(bez(p0, p1, p2, p3, i / SAMPLES_PER_SEG))
    return cl


def _arc_lengths(cl):
    arcs = [0.0]
    for i in range(len(cl) - 1):
        arcs.append(arcs[-1] + math.hypot(cl[i+1][0]-cl[i][0], cl[i+1][1]-cl[i][1]))
    return arcs


def _nearest_arc_progress(px, py, cl, arcs):
    total = arcs[-1] or 1.0
    min_d, best_idx, best_t = float("inf"), 0, 0.0
    for i in range(len(cl) - 1):
        ax, ay = cl[i]; bx, by = cl[i+1]
        dx, dy = bx-ax, by-ay
        lsq = dx*dx + dy*dy
        t = max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy) / lsq)) if lsq else 0.0
        d = math.hypot(px - (ax+t*dx), py - (ay+t*dy))
        if d < min_d:
            min_d, best_idx, best_t = d, i, t
    arc = arcs[best_idx] + best_t * (arcs[best_idx+1] - arcs[best_idx])
    return arc / total


def build_speed_profiles(n_bins=40):
    """
    Returns dict: source → (bin_centers, median_speed, p25_speed, p75_speed)
    Speed in px/ms.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Accumulate speed samples per bin per source
    buckets: dict[str, list[list]] = {}

    sources_to_load = {"human": DATA_DIR / "human"}
    for s in AGENT_DIRS:
        sources_to_load[s] = DATA_DIR / s

    for src, directory in sources_to_load.items():
        if not directory.exists():
            continue
        trajs = load_trajectories(str(directory))
        bins_data = [[] for _ in range(n_bins)]

        for traj in trajs:
            if not traj.get("completed"):
                continue
            cps = traj.get("control_points", [])
            if len(cps) < 4:
                continue
            result = events_to_arrays(traj.get("events", []))
            if result is None:
                continue
            x, y, t = result

            cl = _centerline_from_cps(cps)
            arcs = _arc_lengths(cl)

            dt = np.diff(t)
            dt = np.where(dt == 0, 1e-3, dt)
            dx = np.diff(x); dy = np.diff(y)
            speed = np.sqrt(dx**2 + dy**2) / dt   # px/ms

            # Arc progress at the midpoint between consecutive events
            for i, spd in enumerate(speed):
                mx = (x[i] + x[i+1]) / 2
                my = (y[i] + y[i+1]) / 2
                arc = _nearest_arc_progress(mx, my, cl, arcs)
                b = min(int(arc * n_bins), n_bins - 1)
                bins_data[b].append(spd)

        buckets[src] = bins_data

    profiles = {}
    for src, bins_data in buckets.items():
        medians, p25s, p75s = [], [], []
        for b in bins_data:
            if len(b) >= 3:
                medians.append(np.median(b))
                p25s.append(np.percentile(b, 25))
                p75s.append(np.percentile(b, 75))
            else:
                medians.append(np.nan)
                p25s.append(np.nan)
                p75s.append(np.nan)
        profiles[src] = (bin_centers * 100, np.array(medians),
                         np.array(p25s), np.array(p75s))
    return profiles


def plot_speed_profiles(profiles):
    sources_ordered = ["human"] + [s for s in AGENT_DIRS if s in profiles]

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    fig.suptitle("Speed Profile Along Tunnel Arc\n"
                 "Median px/ms at each arc-progress bin  (shaded = IQR)",
                 fontsize=12)

    for src in sources_ordered:
        if src not in profiles:
            continue
        xs, med, p25, p75 = profiles[src]
        color = SOURCE_COLORS.get(src, "#888")
        lw = 2.5 if src == "human" else 1.8
        ls = "-"  if src == "human" else "--"
        ax.plot(xs, med, color=color, lw=lw, ls=ls,
                label=SOURCE_LABELS.get(src, src))
        ax.fill_between(xs, p25, p75, color=color, alpha=0.12)

    ax.set_xlabel("Arc progress (%)", fontsize=11)
    ax.set_ylabel("Speed (px / ms)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.8)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading feature dataframe...")
    df = load_df()

    print("\nPlotting feature distributions...")
    fig1 = plot_feature_distributions(df)

    print("Computing speed profiles along arc (may take ~60s)...")
    profiles = build_speed_profiles(n_bins=40)

    fig2 = plot_speed_profiles(profiles)

    out1 = OUT_DIR / "plot_feature_distributions.png"
    out2 = OUT_DIR / "plot_speed_profiles.png"
    fig1.savefig(out1, dpi=150)
    fig2.savefig(out2, dpi=150)
    print(f"Saved → {out1}")
    print(f"Saved → {out2}")
    plt.show()


if __name__ == "__main__":
    main()
