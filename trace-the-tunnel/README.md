# Trace-the-Tunnel

A browser game platform for collecting and analyzing mouse trajectory data to study behavioral differences between human and AI agent cursor movements.

## Overview

Players (human or AI) drag a cursor from a green dot to a red dot through a curved tunnel on a 600×350 canvas. Every mouse event is recorded. The resulting trajectory data feeds a classification pipeline that asks: **can a model reliably tell a human trace from an agent trace?**

The game uses a fixed pool of 10 tunnels (seeded, deterministic) so human and agent sessions are directly comparable across runs.

## Repository Layout

```
trace-the-tunnel/
├── server.py                  # Flask game server and trajectory API
├── game.html                  # Game UI (served at /)
├── tunnel_config.json         # Fixed pool of 10 tunnel seeds
├── human_cursor.py            # Human-like cursor path generator
├── CLAUDE.md                  # Instructions for Claude Code as the agent player
│
├── static/
│   ├── game.js                # Game logic and __tunnelGame JS API
│   └── style.css
│
├── collect_agent_data.sh      # Run Claude Code to collect agent trajectories
├── setup.sh                   # Create venv and install dependencies
├── test_one_tunnel.sh         # Quick smoke-test for a single tunnel
│
├── experiments/
│   ├── run.sh                 # Generic experiment harness
│   └── prompts/               # Prompt templates per experiment type
│       ├── agent_minimal.txt
│       └── agent_minimal_fewshot.txt
│
├── analysis/
│   ├── features.py            # Feature extraction (kinematics, geometry, timing)
│   ├── classify.py            # Train/eval classifiers (LR, RF, GBT)
│   ├── classify_fewshot.py    # Few-shot classification variant
│   ├── sequence_model.py      # 1D-CNN classifier on raw sequences
│   ├── state_space.py         # State-space density analysis
│   ├── visualize.py / vis.py  # Trajectory overlay and distribution plots
│   ├── count.py               # Dataset summary utility
│   ├── features.csv           # Cached feature matrix
│   ├── results.json           # Latest classifier results
│   └── plots/                 # Generated figures
│
└── data/
    ├── human/                 # Human session JSONs
    ├── agent/                 # Naive agent session JSONs
    ├── agent_humanlike/       # Human-like agent session JSONs
    └── logs/                  # Claude Code session logs
```

## Setup

```bash
./setup.sh
```

This creates a `.venv` and installs: `flask`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `torch`.

## Running the Game Server

```bash
source .venv/bin/activate
python server.py              # default mode — saves by client-supplied 'source' field
python server.py --experiment agent_minimal  # experiment mode — forces all saves to data/agent_minimal/
```

The server runs at `http://localhost:5050` (configurable with `--port`).

**Experiment mode** enforces data integrity: the server overwrites the `source` field on every save and refuses `source='human'`, so agent trajectories cannot accidentally pollute the human folder.

## Collecting Human Data

Open `http://localhost:5050` in a browser (no `--experiment` flag on the server). Play the game — drag from the green dot to the red dot through the tunnel. Session data is auto-saved to `data/human/` on completion or failure.

## Collecting Agent Data

Requires the [Claude Code CLI](https://claude.ai/code) (`claude`) and a running server.

```bash
# 1 round of naive agent (all 10 tunnels)
./collect_agent_data.sh

# 3 rounds, human-like mode (uses human_cursor.py)
./collect_agent_data.sh --human-like --rounds 3
```

Modes:

| Flag | Source label | Description |
|------|-------------|-------------|
| _(none)_ | `agent` | Naive: Claude moves through the centerline directly |
| `--human-like` | `agent_humanlike` | Uses `human_cursor.py` to add natural movement characteristics |

Each round is a single Claude Code invocation that plays all 10 tunnels sequentially. Logs land in `data/logs/`.

### Running a Named Experiment

Use `experiments/run.sh` for reproducible, named experiment runs. Each experiment has a prompt file and a dedicated output folder:

```bash
# Start server in experiment mode first:
source .venv/bin/activate && python server.py --experiment agent_minimal

# Then in another terminal:
./experiments/run.sh agent_minimal --rounds 3

# With few-shot example images:
./experiments/run.sh agent_minimal_fewshot --examples experiments/examples/example-1.png
```

The harness validates that the server is running in the matching mode, writes a manifest JSON (prompt hash, git SHA, example checksums) alongside the trajectories, and warns if any files leak into `data/human/` during the run.

## Human-Like Cursor Generator

`human_cursor.py` transforms a raw centerline into a trajectory that mimics human motor behavior:

- **Bezier smoothing** — B-spline resampling for smooth curvature
- **Overshoots** — small perpendicular deviations at random points along the path
- **Perpendicular noise** — Gaussian jitter in the direction normal to the path
- **Minimum-jerk timing** — speed profile that is slow at start/end and fast in the middle, matching human reaching movements

```bash
source .venv/bin/activate && python human_cursor.py \
  --centerline '[[x,y], ...]' \
  --noise 2.5 \
  --overshoots 2 \
  --duration 900 \
  --points 150
```

Output: JSON array of `{x, y, delay_ms}` waypoints for Playwright replay.

## Trajectory Data Format

Each session JSON saved to `data/<source>/` contains:

```json
{
  "session_id": "...",
  "tunnel_id": 3,
  "source": "human",
  "completed": true,
  "fail_reason": null,
  "control_points": [...],
  "events": [
    { "event_type": "mousemove", "x": 42.1, "y": 175.3, "timestamp": 1234567890 },
    ...
  ]
}
```

## Analysis Pipeline

All scripts are in `analysis/`. Run from the repo root with the venv active.

### Feature Extraction

```bash
python analysis/features.py
```

Extracts 25 features per trajectory and writes `analysis/features.csv`:

- **Kinematics**: speed, acceleration, jerk (mean, std, max, median)
- **Curvature**: mean, std, max curvature along the path
- **Timing**: inter-event intervals (mean, std, min, max)
- **Geometry**: path length, path efficiency (straight-line / arc), direction changes
- **Centerline deviation**: mean and std distance from the tunnel centerline
- **Frequency**: tremor-band power (8–12 Hz, physiological hand tremor range)

### Classification

```bash
python analysis/classify.py
```

Trains Logistic Regression, Random Forest, and Gradient Boosting classifiers with 5-fold cross-validation to distinguish `human` (label 0) from all agent sources (label 1). Results saved to `analysis/results.json`.

Current results (58 human, 16 agent trajectories):

| Classifier | Accuracy | ROC AUC |
|-----------|----------|---------|
| Logistic Regression | 0.676 | 0.752 |
| Random Forest | 0.622 | 0.724 |
| Gradient Boosting | 0.622 | 0.642 |

Top discriminating features: `accel_mean`, `centerline_dev_mean`, `path_efficiency`, `centerline_dev_std`, `path_length`.

### Sequence Model

```bash
python analysis/sequence_model.py
```

A 1D-CNN that operates directly on raw trajectory sequences (x, y, speed, acceleration, curvature) at fixed length 200, bypassing hand-crafted features.

### Visualization

```bash
python analysis/visualize.py   # trajectory overlays per tunnel
python analysis/vis.py         # feature distributions, speed profiles, state-space density
```

Output figures land in `analysis/plots/`.

## The `__tunnelGame` JavaScript API

The game exposes a global `window.__tunnelGame` object for programmatic control (used by agents via `browser_evaluate`):

```js
window.__tunnelGame.loadTunnel(id)      // load tunnel 0-9
window.__tunnelGame.getTunnelId()
window.__tunnelGame.getStartPos()       // {x, y} in canvas coords
window.__tunnelGame.getEndPos()
window.__tunnelGame.getCenterline()     // array of {x, y}
window.__tunnelGame.getTunnelWidth()
window.__tunnelGame.getCanvasRect()     // DOMRect of the canvas element
window.__tunnelGame.getState()          // "idle" | "playing" | "done_success" | "done_fail"
window.__tunnelGame.getSessionData()    // full session object ready to POST
```

Canvas coordinates (0–600 × 0–350) convert to screen coordinates via:
```
screenX = canvasRect.left + (canvasX / 600) * canvasRect.width
screenY = canvasRect.top  + (canvasY / 350) * canvasRect.height
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve `game.html` |
| `GET` | `/api/tunnels` | Return full tunnel config (IDs + seeds) |
| `GET` | `/api/mode` | Return current experiment mode |
| `POST` | `/api/save_trajectory` | Save a session JSON to `data/<source>/` |
