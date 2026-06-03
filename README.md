# Game-Based CAPTCHA: Measuring the Human–Agent Behavioral Gap

Recognition-based CAPTCHAs have collapsed under multimodal LLMs, which now match or beat
humans on distorted text and image-selection challenges. So we stop asking *whether* a
short web game is solved and look at *how* it is played: the `(x, y, t)` cursor stream a
player leaves behind is a behavioral fingerprint. This repo asks whether that fingerprint
is a durable human–agent separator, and how much of it an adversary can reproduce.

> See [`final_report.pdf`](final_report.pdf) for the full write-up.

## What we asked, and what we found

- **Q1 — Does trajectory information add separation beyond outcome alone?**
  **Yes.** On the motor game the trajectory classifier reached **AUC 1.0** against every
  agent that had not seen the exact path before, while outcome (did it solve the game) was
  no help at all.
- **Q2 — Is the gap robust to a basic imitation attack?**
  **Yes.** Few-shot LLM solvers, ~30 min of human-guided iteration, and a trained RL policy
  all failed to drive the detector toward chance on unseen tunnels.

The hypothesis held: humans leave structured, continuous motor noise — physiological
tremor and real-time error correction — while an agent that synthesizes events
programmatically produces trajectories that are more uniform, faster, and free of any
involuntary biological signal, whether or not it solves the game. No attacker drove the
detector to bypass detection on tunnels it had not seen.

## The games

| Game | Channel | What the player does |
|------|---------|----------------------|
| **Trace-the-Tunnel** (`trace-the-tunnel/`) | Motor | Drag from a green dot to a red dot along a curved tunnel on a 600×350 canvas without crossing the walls. Leaving the boundary auto-resets. ~1.3 s for humans. |
| **Pick-the-Personality** (`pick-the-personality/`) | Cognitive | Four dots move with distinct behavioral signatures (leader, follower, hyper, lazy); click the one matching a named personality. |
| **Roll-or-Slide** (`roll-or-slide/`) | Cognitive | Four shapes move; click the one whose motion follows a different rule. Six variations (`game.html`, `game_v2`…`game_v6`). |

Trace-the-Tunnel is the only game that produces a usable trajectory: it forces a dense,
high-frequency record of *how* a cursor is driven through a constrained path. The
reasoning games are a deliberate contrast — see [Findings](#findings).

## Approach

**Attacker model.** Rather than a fixed bot or an offline trajectory generator, the
adversary is a live coding agent (Claude Code) given the full attacker surface: it drives
the browser through Playwright, inspects each game's source, and writes and executes its
own solver scripts in a tight loop. A small **RL agent** (behavioral cloning + PPO over a
Python reimplementation of the tunnel, dispatching via the Chrome DevTools Protocol) is
also evaluated.

**Trajectory representation.** Each Trace-the-Tunnel session is logged as raw
`mousemove`/`mousedown`/`mouseup` events `(x, y, t)` timestamped with `performance.now()`,
with no throttling, so inter-event intervals carry the timing signature of the input
pipeline. We summarize each trajectory with **25 features in 6 groups** (kinematics,
curvature, path geometry, centerline deviation, 8–12 Hz tremor, inter-event timing), split
into two heads:

- **Motor head** (20 features) — *how the hand moved*
- **Mechanism head** (5 timing features) — *how the events were produced*

**Detector.** A RandomForest (200 trees, seed 42, on standardized features). Cross-checked
against logistic regression (AUC 0.999), gradient boosting (0.984), and a feature-free
1D-CNN over the raw event sequence (0.985), so the signal does not depend on the chosen
features. The detector is treated as **holding** above AUC 0.85 and **defeated** below;
AUC ≈ 0.5 is the human-indistinguishable region.

## Experiments and results

Collected on a single Apple-silicon MacBook Pro: **217 human sessions across 6
participants** on a shared pool of 10 tunnels, plus 30–120 agent sessions per strategy.

| Experiment | Question | Result |
|------------|----------|--------|
| **1 — Replay** (`experiments_replay_{cdp,hid,init}/`) | Can the agent deliver a real human trace back indistinguishably? | Only by forging the Web APIs the captcha reads. CDP → AUC 0.82, HID → 0.63, init-forge → **0.36**. No userspace dispatch channel is leak-free. |
| **2 — Few-shot generation** (`experiments_generalize/`) | Can the agent generate traces for *unseen* tunnels? | No. Opus/Sonnet/Haiku all stay at **AUC 1.0** — paths hug the centerline and speed peaks cluster at the wrong times. |
| **3 — Guided iteration** | With active human guidance, can it do better? | Marginally: ~30 min/tunnel buys 1.0 → **0.998**. Solvers are structurally bad at expressing contextual human preference. |
| **4 — RL agent** (`experiments_rl/`) | Can a trained policy beat LLM-written solvers? | No. BC+PPO reaches **AUC 1.0** with the same flat speed profile; the failure is not specific to one technology. |

**Where the signal lives.** Once delivery is faithful, the motor head alone returns
AUC 1.0. Three features carry it, and all three are about the *player*, not the path:

1. **Speed-peak location** — agents apply one deterministic policy, so peaks cluster; humans react in real time, so peaks scatter.
2. **Risk profile** — pushed toward human-looking features, agents graze walls a human would never risk, having no internal cost for the reset.
3. **Continuous-correction noise** — humans produce involuntary 8–12 Hz tremor; computed paths have no power in that band.

RandomForest importances confirm the classifier reads exactly these: centerline-deviation
mean (0.23) and std (0.22), inter-event Δt mean (0.11), speed std (0.08).

**Reasoning-game contrast.** Pick-the-Personality and Roll-or-Slide produce no trajectory,
so there is no AUC to report. On outcome metrics the gap is small or absent even though the
two processes are completely different (a human glances and pattern-matches; the agent
captures a screenshot stack and computes lagged cross-correlations). The lesson is a design
principle: **the discriminating signal must be a necessary byproduct of completing the
task** — a single-commit click discards everything between observation and answer.

## Repository layout

```
trace-the-tunnel/          motor game + the bulk of the analysis
├── game.html, server.py   Flask server; static/game.js renders the tunnel and logs events
├── static/                game.js, centerline/waypoints/path JSON
├── analysis/              features.py, classify.py, two_head_eval.py, sequence_model.py, plots/
├── data/                  human/, visual/, replay_*/, gen_*/ (per-model), rl_agent/
├── experiments_replay_cdp|hid|init/   Experiment 1 (delivery channel)
├── experiments_generalize/            Experiment 2 (unseen-tunnel generation)
└── experiments_rl/        Experiment 4: bc_train.py, ppo_train.py, tunnel_env.py, policy.py
pick-the-personality/      reasoning game (cognitive channel)
roll-or-slide/             reasoning game, six variations (game.html … game_v6.html)
final_report.pdf           full write-up
Checkpoint1.md, Checkpoint2.md, checkpoint_evals.md   milestone notes
```

## Running it

Each game is a self-contained Flask app. For Trace-the-Tunnel:

```bash
cd trace-the-tunnel
./setup.sh                       # one-time: create .venv, install deps
source .venv/bin/activate
python server.py                 # default mode — saves by client-supplied 'source'
```

The server runs at `http://localhost:5050`. Open it in a browser and play; sessions
auto-save to `data/human/` on completion or failure.

**Experiment mode** locks the output folder for a named agent run, overwriting the `source`
field on every save and refusing `source='human'` so agent traces can't pollute the human
dataset:

```bash
python server.py --experiment agent_minimal
```

See [`trace-the-tunnel/README.md`](trace-the-tunnel/README.md) for collecting agent data
with Claude Code (run it from a *separate, empty directory* — the agent behaves differently
if it can read the game source). The `pick-the-personality/` and `roll-or-slide/` games each
have their own `setup.sh` and `server.py`.

## Takeaways

Motor games show real potential for separating humans from both LLM coding agents and RL
agents: in every condition where the agent had not seen the exact path before, the
trajectory classifier reached **AUC 1.0**. The asymmetry favors the defender — an attacker
pays ~30 minutes of guided iteration per tunnel to recover a fraction of a percent of
detector AUC, while a defender who complicates the geometry or adds moving obstacles
invalidates all of that work for free.

## Limitations

A deliberately narrow systems study. The human data are small (217 sessions, 6 Stanford CS
students, all on MacBooks in Chromium), so the classifier may partly fit this setting; we
did not vary device, OS, or input modality. The adversary is an LLM coding agent plus one
RL policy — **not** a vision-language agent that perceives the rendered game end-to-end, so
conclusions are scoped to LLM and RL agents, and VLM agents remain the most relevant
untested threat. Finally, clean measurement depends on a fixed tunnel pool, which is also
what an adversary can memorize: rotating or expanding the pool trades measurement
cleanliness for robustness.
