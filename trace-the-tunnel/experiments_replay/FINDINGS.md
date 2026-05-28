# Replay experiment — setup and findings

Run dates: 2026-05-25
Git SHA at v1 run: 581c2f59fd80f63f4f11735ebad905d578596348

## Motivation

Checkpoint 1 evaluated a visual-perception agent that synthesised a path
through each tunnel (centerline + sinusoidal swerve, dispatched via
`browser_evaluate`). It was distinguishable from humans at AUC = 1.000.
The largest tells were spatial:

- `centerline_dev_mean` — 0.229 RF importance
- `centerline_dev_std`  — 0.221 RF importance
- `dt_mean`             — 0.110 RF importance
- `speed_std`           — 0.079 RF importance

The hypothesis was that a *human-trajectory replay* agent — one that
samples a real completed human trace for the live tunnel and dispatches
the original `(x, y, t)` events back into the canvas — would collapse
those spatial tells, because the trajectory literally *is* a human
trajectory inside the same tunnel.

## Variants run so far

| Name        | Bank source                                                  | n_replays | Purpose |
|-------------|--------------------------------------------------------------|----------:|---------|
| `replay`    | Full pool of all 172 completed humans (overlaps eval set)    | 86 | Initial run; numbers are tainted by source-set leakage and are kept only for ablation. |
| `replay_v1b` | 3 humans per tunnel (30 total), pre-selected and held out of the eval-time human set | 84 | **Canonical v1 result.** Strictly disjoint source / eval split. |

The v1 run is reported below only as a baseline for ablating what the
disjoint protocol changes. All headline numbers reference `replay_v1b`.

## Experiment setup

### Why no spatial warping is needed

The tunnel pool is deterministic (`tunnel_config.json` fixes 10 seeds).
Every completed human trace for a given `tunnel_id` has identical
`control_points`. Verified empirically on all 10 tunnels: 16–18
completed human traces per tunnel, 1 unique control-point set per
tunnel. The replay can therefore dispatch raw human `(x, y)` directly
into the same tunnel and stay inside by construction.

### Server changes

Added two endpoints / flags to `server.py`:

```
GET /api/human_bank/<tunnel_id>
  → list of {session_id, tunnel_id, events: [...]}
    for every completed trace in data/human/ matching tunnel_id

CLI: --allowed-sessions <file.json>
  → when set, /api/human_bank only returns sessions whose session_id
    is in the JSON allowlist. Used to enforce the v1b disjoint split.
```

### Disjoint source/eval split (v1b)

The concern with the original `replay` run was source-set leakage: the
classifier's "human" class was the full pool of 172 humans, and the
replay agent sampled from that same pool. A held-out human in CV could
be the *source* of a replay in another fold. To rule this out:

1. `experiments_replay/select_bank.py --per-tunnel 3 --seed 42` picks 3
   completed humans per tunnel (30 total). Selection is seeded.
2. The chosen session_ids are written to
   `experiments_replay/allowed_sessions_v1b.json` (consumed by the
   server) and `..._meta.json` (human-readable manifest of the bank:
   seed, tunnel coverage, source filenames, event counts).
3. Server is launched with `--allowed-sessions <file>` so
   `/api/human_bank/<tid>` returns only those 30. The replay solver is
   physically prevented from sampling anything else.
4. The eval (`experiments_replay/eval_disjoint.py`) loads the same
   allowlist and removes those 30 session_ids from the human class
   before training. Eval-time humans: 142. Bank-only humans: 30.
5. Reproducibility check: the eval script verifies that no replay
   trace's session_id collides with a bank session_id (should be 0).

### Agent harness

New sibling folder `experiments_replay/` (leaves the original
`experiments/` visual-agent harness untouched):

- `replay_solver.js` — IIFE that installs `window.__replay`. For each
  tunnel: fetch the bank, sample one trace, dispatch its events in
  order with `setTimeout(dt_source)` between each `MouseEvent`.
  Early-breaks when `window.__tunnelGame.getState()` reaches a
  terminal state. Up to 4 internal retries per tunnel.
- `prompts/replay_v1b.txt` — instructs Claude Code to navigate, Read
  the solver, inject it via `browser_evaluate`, and call
  `await window.__replay.runAll(3)`.
- `run.sh` — generic harness; takes the experiment name as its first
  argument, records the solver SHA in the per-run manifest at
  `data/<name>/manifest_<ts>.json`.

Saves are forced into `data/replay_v1b/` by running the server with
`--experiment replay_v1b`, which also rejects `source: 'human'`.

### Run

```bash
# Terminal A
source .venv/bin/activate
python server.py --experiment replay_v1b \
  --allowed-sessions experiments_replay/allowed_sessions_v1b.json

# Terminal B
./experiments_replay/run.sh replay_v1b --rounds 3
```

3 rounds × 10 tunnels × ≤3 attempts. Manifest:
`data/replay_v1b/manifest_20260525_140152.json`.

### Data collected

- 84 successful `replay_v1b/` trajectories written.
- Completion rate sits in the same range as humans (92%) and v1 (93%);
  not a tell.

## Headline result

| Setting                           | LR AUC | RF AUC | GB AUC |
|-----------------------------------|-------:|-------:|-------:|
| Baseline (human vs visual)        | 1.000  | **1.000** | 0.987  |
| **Replay v1b — disjoint, 3/tunnel** | 0.994 | **0.998** | 0.989 |
| Replay v1 — same protocol, full bank (leaky) | 0.974 | 0.996 | 0.994 |

Eval-time class sizes: human = 142, replay_v1b = 84, visual = 122.

Replay does not pass the classifier. The disjoint protocol confirms the
v1 result was not an artifact of source-set leakage: removing the 30
bank humans from the eval class moves RF AUC from 0.996 to 0.998
(slightly *up*, not down).

## What the replay killed

Per-source means on the four features that drove the baseline classifier:

| Feature                | Human  | Visual | Replay v1 | Replay v1b |
|------------------------|-------:|-------:|----------:|-----------:|
| `centerline_dev_mean`  | 11.05  | 17.73  | 11.20     | **10.49** |
| `centerline_dev_std`   |  6.76  | 10.21  |  6.94     | **6.48**  |
| `direction_changes`    |  8.1   | 23.6   |  6.85     |  7.46     |
| `speed_mean`           |  0.49  |  0.64  |  0.45     |  **0.46** |
| `speed_std`            |  0.24  |  0.15  |  0.20     |  0.21     |

The spatial / kinematic features that accounted for ~64% of the
baseline RF's importance are now indistinguishable between replay v1b
and human at the population level. centerline_dev — the single largest
tell — is matched to within ~5% on both mean and std. The disjoint
constraint did not weaken this result; if anything v1b is slightly
*closer* to the human distribution than v1 was.

## What the replay newly leaked

RF feature importance shifted almost entirely onto timing:

| Rank | Baseline (human vs visual) | Replay v1b (disjoint)     |
|------|----------------------------|--------------------------|
| 1 | centerline_dev_mean (0.260) | **dt_mean (0.391)** |
| 2 | centerline_dev_std (0.241)  | jerk_std (0.107) |
| 3 | dt_mean              (0.090) | **dt_std  (0.103)** |
| 4 | speed_std            (0.076) | dt_max   (0.046) |
| 5 | speed_median         (0.033) | accel_std (0.030) |

Per-event `dt` distribution for `mousemove` events (computed from v1
run — re-running for v1b expected to be similar in shape, larger in
spread):

| Source | n | mean | p5 | p25 | p50 | p75 | p95 | frac < 4 ms |
|--------|---:|---:|---:|---:|---:|---:|---:|---:|
| human  | 27,103 | 8.46 | 7.5 | 8.3 | 8.3 | 8.4 | 9.3 | 0.00% |
| replay (v1) | 13,715 | 9.13 | 8.2 | 8.6 | 9.0 | 9.5 | 10.0 | 0.00% |
| visual | 18,537 | 7.26 | 4.5 | 5.2 | 7.1 | 9.1 | 10.8 | 0.00% |

Per-trace `dt_std` (mean over sessions, from the disjoint eval):

| Source     | per-trace dt_std |
|------------|-----------------:|
| human      | 0.94 |
| replay (v1)  | 1.06 |
| replay v1b | **1.81** |
| visual     | 0.69 |

The 4 ms `setTimeout` clamp is *not* the culprit — neither humans nor
replay drop below 5 ms. The mean shift is a per-call dispatch overhead
(`setTimeout` macrotask scheduling + `MouseEvent` construction +
`dispatchEvent`), which the OS input loop doesn't pay per event. With
the original v1 bank of 17 per tunnel, this jitter averaged out across
many source samples and `dt_std` looked nearly human (1.06 vs 0.94).
At v1b's 3 per tunnel, a single source trace's per-event jitter
dominates each session and `dt_std` widens to 1.81 — almost 2×
human. The smaller bank made `dt_std` a top-3 feature for the
classifier; in v1 it didn't crack the top 5.

This is the inverse of what you might intuitively expect ("fewer
sources = harder for the classifier"). Larger source banks let scheduler
noise wash out across samples; smaller banks expose it.

## Interpretation

The baseline classifier exploited a *content* difference — the agent's
trajectory shape didn't match a human's. The replay classifier
exploits a *mechanism* difference — the agent's event scheduler has a
sub-millisecond fingerprint that no human input device has. The attack
surface narrowed from six feature groups to essentially three timing
features (dt_mean / dt_std / dt_max), with `dt_mean` dominating.

The disjoint-source protocol confirms this isn't an artifact of how we
held out the eval set: even when the classifier has provably never seen
the source traces the agent drew from, the dt fingerprint is enough to
separate the populations at RF AUC = 0.998.

This is a substantively different result for the writeup: it shifts
the captcha's effective discriminator from "the agent reasons about
motion differently" to "the agent dispatches events differently."
That's a much narrower thing to defend, a much narrower thing for the
attacker to forge from inside the page, and — crucially — invariant to
how diverse our human population is.

## Open questions for the next variant

- **v1c — radical bank shrinkage.** Drop the bank to ~5 total source
  traces (stratified to maximize tunnel coverage). The agent will only
  be able to play the tunnels covered. Two predictions: (1) `dt_std`
  widens further as the per-trace jitter loses any chance to average;
  (2) the per-tunnel replay output becomes near-deterministic, which
  may itself become a tell if the classifier sees enough repetitions.
- **v2 — better dispatch primitive.** Swap `setTimeout` for
  `MessageChannel` + closed-loop absolute-time scheduling. Tests
  whether the residual signal is the scheduler choice or something
  deeper about in-page dispatch.
- **v3 — channel escalation.** Move dispatch out of page JS to
  Playwright's CDP-level `Input.dispatchMouseEvent`. If the mechanism
  tell vanishes here, the captcha's effective defense is channel-bound,
  not behavior-bound.

## Files

- Code: `experiments_replay/replay_solver.js`, `select_bank.py`,
  `eval_disjoint.py`, `prompts/replay_v1b.txt`, `run.sh`
- Server: `server.py` (`/api/human_bank/<tunnel_id>` +
  `--allowed-sessions <file>` flag)
- v1b allowlist: `experiments_replay/allowed_sessions_v1b.json` +
  `..._meta.json`
- Data: `data/replay/` (v1, 86 trajectories), `data/replay_v1b/` (84
  trajectories)
- Numbers (machine-readable): `analysis/replay_v1b_eval.json`
