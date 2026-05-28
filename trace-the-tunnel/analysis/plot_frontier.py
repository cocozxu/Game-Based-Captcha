"""
Scatter plot of motor-AUC vs mechanism-AUC for every attack.

Reads analysis/two_head_eval.json. Each attack is one point; (0.5, 0.5) is the
human-equivalent region (classifier can't distinguish). Far from (0.5, 0.5) =
attack is detectable on that axis. Combined AUC is annotated next to each point.

Run:
  python analysis/plot_frontier.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# Group attacks by family for color coding.
FAMILIES = {
    "no-prior": {"members": ["visual"], "color": "#d32f2f", "marker": "X"},
    "replay": {"members": ["replay", "replay_v1b", "replay_v1c", "replay_v2"], "color": "#1976d2", "marker": "o"},
    "replay-timed (busy-wait dispatch)": {"members": ["replay_timed"], "color": "#f57c00", "marker": "D"},
    "warp":    {"members": ["warp_v1", "warp_t1"], "color": "#7b1fa2", "marker": "s"},
}


def family_of(name):
    for fam, info in FAMILIES.items():
        if name in info["members"]:
            return fam
    return "other"


def main():
    with open(os.path.join(HERE, "two_head_eval.json")) as f:
        data = json.load(f)

    points = []  # (name, motor, mech, combined, family)
    for name, info in data["attacks"].items():
        if info.get("skipped"):
            print(f"[skip] {name}: {info.get('reason')}")
            continue
        h = info["heads"]
        points.append((
            name,
            h["motor"]["auc"],
            h["mechanism"]["auc"],
            h["combined"]["auc"],
            family_of(name),
        ))

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 8))

    def draw_points(ax, points, name_offsets=None):
        plotted_families = set()
        for name, motor, mech, combined, fam in points:
            info = FAMILIES.get(fam, {"color": "#666", "marker": "v"})
            label = fam if fam not in plotted_families else None
            plotted_families.add(fam)
            ax.scatter(
                motor, mech,
                s=180, c=info["color"], marker=info["marker"],
                edgecolors="black", linewidths=1.0, zorder=3,
                label=label,
            )
            dx, dy = (8, 8)
            if name_offsets and name in name_offsets:
                dx, dy = name_offsets[name]
            ax.annotate(
                f"{name}\n(comb={combined:.3f})",
                xy=(motor, mech),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"),
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.7),
                zorder=4,
            )

    # -------- LEFT PANEL: full picture (shows human region is empty) --------
    ax_full.add_patch(plt.Rectangle((0.45, 0.45), 0.10, 0.10, color="#4caf50", alpha=0.4, zorder=0))
    ax_full.text(0.5, 0.5, "human\nregion\n(undetectable)",
                 ha="center", va="center",
                 fontsize=10, fontweight="bold", color="#1b5e20", zorder=1)
    ax_full.plot([0.5, 1.0], [0.5, 1.0], "--", color="gray", alpha=0.4, zorder=1, label="y = x")
    ax_full.axhline(0.95, color="gray", linestyle=":", alpha=0.3, zorder=1)
    ax_full.axvline(0.95, color="gray", linestyle=":", alpha=0.3, zorder=1)

    # Box showing where the zoom panel is. Expand to include replay_timed (~0.94, 0.94).
    zoom_xlim = (0.86, 1.01)
    zoom_ylim = (0.92, 1.005)
    ax_full.add_patch(plt.Rectangle(
        (zoom_xlim[0], zoom_ylim[0]),
        zoom_xlim[1] - zoom_xlim[0], zoom_ylim[1] - zoom_ylim[0],
        fill=False, edgecolor="red", linestyle="-", linewidth=1.5, zorder=2,
    ))
    ax_full.annotate("zoomed →", xy=(zoom_xlim[1], (zoom_ylim[0] + zoom_ylim[1]) / 2),
                     xytext=(-6, 30), textcoords="offset points",
                     fontsize=10, color="red", ha="right")

    draw_points(ax_full, points)
    ax_full.set_xlim(0.45, 1.03)
    ax_full.set_ylim(0.45, 1.03)
    ax_full.set_xlabel("Motor-only AUC  →  detectable from trajectory shape", fontsize=11)
    ax_full.set_ylabel("Mechanism-only AUC  →  detectable from event timing", fontsize=11)
    ax_full.set_title("Full picture: replay_timed shifts visibly off the (1, 1) corner", fontsize=11)
    ax_full.grid(True, alpha=0.25)
    ax_full.legend(loc="lower right", framealpha=0.9, title="Attack family", fontsize=9)
    ax_full.text(1.01, 0.52, "ONLY motor catches", ha="right", va="center",
                 fontsize=9, color="#888", style="italic")
    ax_full.text(0.52, 1.01, "ONLY mechanism catches", ha="left", va="center",
                 fontsize=9, color="#888", style="italic", rotation=90)

    # -------- RIGHT PANEL: zoomed corner where all 7 attacks live --------
    ax_zoom.plot([zoom_xlim[0], zoom_xlim[1]],
                 [zoom_xlim[0], zoom_xlim[1]],
                 "--", color="gray", alpha=0.4, zorder=1)  # y = x reference

    # Spread labels out so they don't overlap.
    name_offsets = {
        "visual":       ( 8,  10),
        "replay":       (-90, 18),
        "replay_v1b":   (-90, -28),
        "replay_v1c":   (10, -22),
        "replay_v2":    (10,  22),
        "replay_timed": (10,  10),
        "warp_v1":      (-90,  10),
        "warp_t1":      (-90, -25),
    }
    draw_points(ax_zoom, points, name_offsets)

    ax_zoom.set_xlim(zoom_xlim)
    ax_zoom.set_ylim(zoom_ylim)
    ax_zoom.set_xlabel("Motor-only AUC", fontsize=11)
    ax_zoom.set_ylabel("Mechanism-only AUC", fontsize=11)
    ax_zoom.set_title("Zoom into top-right: per-attack positions", fontsize=11)
    ax_zoom.grid(True, alpha=0.25)
    ax_zoom.legend(loc="lower left", framealpha=0.9, title="Attack family", fontsize=9)

    fig.suptitle(
        "Attack Pareto frontier: motor vs mechanism detectability\n"
        "replay_timed (busy-wait dispatcher) is the first attack to meaningfully move toward the human region.",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    out_path = os.path.join(HERE, "plots", "motor_vs_mechanism_frontier.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"Saved {out_path}")

    # Also write a flat CSV for the writeup table.
    csv_path = os.path.join(HERE, "two_head_eval_table.csv")
    with open(csv_path, "w") as f:
        f.write("attack,family,motor_auc,mechanism_auc,combined_auc,n_human,n_attack\n")
        for name, motor, mech, combined, fam in points:
            info = data["attacks"][name]
            f.write(f"{name},{fam},{motor:.4f},{mech:.4f},{combined:.4f},"
                    f"{info['n_human']},{info['n_attack']}\n")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
