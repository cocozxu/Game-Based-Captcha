"""
State-space density analysis: compare which regions of
(x, y, speed, curvature) space human vs agent trajectories occupy.

Produces:
  - 2D density heatmaps (x vs y, speed vs curvature, etc.)
  - KL divergence between human and agent state distributions
  - State-space occupancy overlap metrics

Run: python analysis/state_space.py
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import rel_entr

from features import load_trajectories, events_to_arrays

matplotlib.rcParams["figure.dpi"] = 120

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Extract state vectors from trajectories
# ---------------------------------------------------------------------------

def extract_states(trajs):
    """
    Extract (x_norm, y_norm, speed, curvature) from a list of trajectories.
    Returns a (N, 4) array where N is total number of points across all trajs.
    """
    all_states = []

    for traj in trajs:
        if not traj.get("completed"):
            continue
        result = events_to_arrays(traj.get("events", []))
        if result is None:
            continue

        x, y, t = result
        if len(x) < 10:
            continue

        canvas = traj.get("canvas_size", {"width": 600, "height": 350})
        x_norm = x / canvas["width"]
        y_norm = y / canvas["height"]

        # Speed
        dt = np.diff(t)
        dt = np.where(dt == 0, 1e-3, dt)
        speed = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt
        speed = np.concatenate([[0], speed])

        # Curvature
        gx = np.gradient(x)
        gy = np.gradient(y)
        ggx = np.gradient(gx)
        ggy = np.gradient(gy)
        denom = (gx**2 + gy**2)**1.5
        denom = np.where(denom == 0, 1e-9, denom)
        kappa = np.abs(gx * ggy - gy * ggx) / denom

        states = np.stack([x_norm, y_norm, speed, kappa], axis=1)
        all_states.append(states)

    if not all_states:
        return np.empty((0, 4))
    return np.concatenate(all_states, axis=0)


def load_states_by_source():
    """Load states grouped by source type."""
    source_states = {}
    for source_dir in sorted(os.listdir(BASE_DIR)):
        full = os.path.join(BASE_DIR, source_dir)
        if not os.path.isdir(full) or source_dir == "logs":
            continue
        trajs = load_trajectories(full)
        states = extract_states(trajs)
        if len(states) > 0:
            source_states[source_dir] = states
    return source_states


# ---------------------------------------------------------------------------
# KL divergence on binned distributions
# ---------------------------------------------------------------------------

def binned_kl_divergence(states_a, states_b, dims, n_bins=30, clip_range=None):
    """
    Compute KL(A || B) on a 2D histogram of the given dimension indices.
    Returns KL divergence and the two histograms.
    """
    a = states_a[:, dims]
    b = states_b[:, dims]

    if clip_range is not None:
        ranges = clip_range
    else:
        combined = np.concatenate([a, b], axis=0)
        ranges = [(np.percentile(combined[:, i], 1), np.percentile(combined[:, i], 99))
                   for i in range(len(dims))]

    hist_a, edges = np.histogramdd(a, bins=n_bins, range=ranges, density=True)
    hist_b, _ = np.histogramdd(b, bins=n_bins, range=ranges, density=True)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    hist_a = hist_a + eps
    hist_b = hist_b + eps

    # Normalize to proper distributions
    hist_a = hist_a / hist_a.sum()
    hist_b = hist_b / hist_b.sum()

    # KL(A || B)
    kl = np.sum(rel_entr(hist_a, hist_b))

    return kl, hist_a, hist_b, edges


def occupancy_overlap(states_a, states_b, dims, n_bins=30):
    """
    Compute the fraction of bins occupied by both A and B.
    A bin is "occupied" if it has > 0.1% of that source's points.
    """
    a = states_a[:, dims]
    b = states_b[:, dims]
    combined = np.concatenate([a, b], axis=0)
    ranges = [(np.percentile(combined[:, i], 1), np.percentile(combined[:, i], 99))
               for i in range(len(dims))]

    hist_a, _ = np.histogramdd(a, bins=n_bins, range=ranges)
    hist_b, _ = np.histogramdd(b, bins=n_bins, range=ranges)

    thresh_a = len(a) * 0.001
    thresh_b = len(b) * 0.001

    occ_a = hist_a > thresh_a
    occ_b = hist_b > thresh_b

    overlap = (occ_a & occ_b).sum()
    union = (occ_a | occ_b).sum()

    iou = overlap / union if union > 0 else 0
    return iou


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_2d_density(states_dict, dim_x, dim_y, xlabel, ylabel, filename, n_bins=50):
    """
    Side-by-side 2D density heatmaps for each source.
    """
    sources = sorted(states_dict.keys())
    n = len(sources)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 4.5))

    # Compute shared range
    all_vals = np.concatenate(list(states_dict.values()), axis=0)
    x_range = (np.percentile(all_vals[:, dim_x], 1), np.percentile(all_vals[:, dim_x], 99))
    y_range = (np.percentile(all_vals[:, dim_y], 1), np.percentile(all_vals[:, dim_y], 99))

    colors = {"human": "Blues", "agent": "Oranges", "agent_humanlike": "Purples"}

    hists = {}
    for i, src in enumerate(sources):
        ax = axes[i]
        s = states_dict[src]
        h, xedges, yedges = np.histogram2d(
            s[:, dim_x], s[:, dim_y], bins=n_bins, range=[x_range, y_range], density=True
        )
        hists[src] = h
        ax.imshow(h.T, origin="lower", aspect="auto",
                  extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                  cmap=colors.get(src, "Greys"))
        ax.set_title(f"{src} (n={len(s):,})")
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)

    # Difference map (if exactly 2 sources, or human vs first agent)
    ax_diff = axes[-1]
    if "human" in hists and len(hists) >= 2:
        agent_key = [k for k in hists if k != "human"][0]
        h_human = hists["human"] / (hists["human"].sum() + 1e-10)
        h_agent = hists[agent_key] / (hists[agent_key].sum() + 1e-10)
        diff = h_human - h_agent
        vmax = max(abs(diff.min()), abs(diff.max()))
        im = ax_diff.imshow(diff.T, origin="lower", aspect="auto",
                            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                            cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax_diff.set_title(f"human - {agent_key}")
        ax_diff.set_xlabel(xlabel)
        plt.colorbar(im, ax=ax_diff, shrink=0.8)
    else:
        ax_diff.set_visible(False)

    fig.suptitle(f"State-Space Density: {xlabel} vs {ylabel}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename))
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_1d_marginals(states_dict, dim, xlabel, filename, n_bins=60):
    """Overlaid 1D histograms for a single state dimension."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"human": "#4fc3f7", "agent": "#ff7043", "agent_humanlike": "#ab47bc"}

    all_vals = np.concatenate([s[:, dim] for s in states_dict.values()])
    lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)

    for src in sorted(states_dict.keys()):
        vals = states_dict[src][:, dim]
        ax.hist(vals, bins=n_bins, range=(lo, hi), alpha=0.5,
                label=f"{src} (n={len(vals):,})", color=colors.get(src, "#888"),
                density=True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(f"State Distribution: {xlabel}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename))
    plt.close(fig)
    print(f"  Saved {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading state vectors...")
    states_dict = load_states_by_source()

    for src, s in states_dict.items():
        print(f"  {src}: {len(s):,} state points")

    if len(states_dict) < 2:
        print("ERROR: Need at least 2 sources. Collect more data!")
        sys.exit(1)

    dim_names = ["x_norm", "y_norm", "speed", "curvature"]

    # --- 2D density plots ---
    print("\nGenerating 2D density plots...")
    plot_2d_density(states_dict, 0, 1, "x (normalized)", "y (normalized)",
                    "state_density_xy.png")
    plot_2d_density(states_dict, 2, 3, "speed (px/ms)", "curvature",
                    "state_density_speed_curvature.png")
    plot_2d_density(states_dict, 0, 2, "x (normalized)", "speed (px/ms)",
                    "state_density_x_speed.png")

    # --- 1D marginals ---
    print("\nGenerating 1D marginal plots...")
    plot_1d_marginals(states_dict, 2, "Speed (px/ms)", "state_marginal_speed.png")
    plot_1d_marginals(states_dict, 3, "Curvature", "state_marginal_curvature.png")

    # --- KL divergence ---
    print("\n" + "=" * 50)
    print("KL DIVERGENCE (human || agent)")
    print("=" * 50)

    sources = sorted(states_dict.keys())
    human_states = states_dict.get("human")
    if human_states is None:
        print("No human data found!")
        sys.exit(1)

    results = {}
    dim_pairs = [
        ([0, 1], "x-y"),
        ([2, 3], "speed-curvature"),
        ([0, 2], "x-speed"),
        ([1, 2], "y-speed"),
    ]

    for agent_src in [s for s in sources if s != "human"]:
        agent_states = states_dict[agent_src]
        print(f"\n  human vs {agent_src}:")

        pair_results = {}
        for dims, name in dim_pairs:
            kl, _, _, _ = binned_kl_divergence(human_states, agent_states, dims)
            iou = occupancy_overlap(human_states, agent_states, dims)
            print(f"    {name:<20} KL={kl:.4f}  IoU={iou:.3f}")
            pair_results[name] = {"kl_divergence": float(kl), "occupancy_iou": float(iou)}

        results[f"human_vs_{agent_src}"] = pair_results

    # Save results
    out_path = os.path.join(OUT_DIR, "..", "state_space_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
