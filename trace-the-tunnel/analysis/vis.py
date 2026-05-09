import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "trace-the-tunnel/data/human"     # folder containing json files
BATCH_SIZE = 10

# ============================================================
# LOAD FILES
# ============================================================

json_files = sorted(Path(DATA_DIR).glob("*.json"))

if not json_files:
    raise RuntimeError(f"No json files found in {DATA_DIR}")

print(f"Found {len(json_files)} json files")

# ============================================================
# VISUALIZE IN BATCHES OF 10
# ============================================================

for batch_start in range(0, len(json_files), BATCH_SIZE):

    batch = json_files[batch_start: batch_start + BATCH_SIZE]

    n = len(batch)

    cols = 5
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))

    # flatten axes safely
    if rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.reshape(-1)

    for ax in axes[n:]:
        ax.axis("off")

    for idx, fp in enumerate(batch):

        ax = axes[idx]

        with open(fp, "r") as f:
            data = json.load(f)

        events = data.get("events", [])

        if not events:
            ax.set_title(fp.name)
            ax.text(0.5, 0.5, "NO EVENTS", ha="center")
            ax.axis("off")
            continue

        xs = [e["x"] for e in events]
        ys = [e["y"] for e in events]

        # plot trajectory
        ax.plot(xs, ys, linewidth=1)

        # start point
        ax.scatter(xs[0], ys[0], s=40, marker="o")

        # end point
        ax.scatter(xs[-1], ys[-1], s=40, marker="x")

        # invert y axis to match screen coords
        ax.invert_yaxis()

        # metadata
        source = data.get("source", "unknown")
        completed = data.get("completed", False)
        violations = data.get("boundary_violations", "?")
        duration = int(data.get("duration_ms", 0))

        ax.set_title(
            f"{fp.name}\n"
            f"src={source} complete={completed}\n"
            f"viol={violations} dur={duration}ms",
            fontsize=8
        )

        ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()
