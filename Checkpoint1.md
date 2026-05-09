# Checkpoint 1

## Problem Statement

We scoped the project to the following question: **if we use a game as a CAPTCHA, can we recognize fundamental differences in how LLM agents and humans reason about the game?** 

## System Constraints

For a game to be useful here, it has to satisfy three constraints:

1. **Reasoning is the only allowed source of difference.** The game cannot rely on real-time reaction speed or absolute time-to-act as a discriminator. Otherwise the test reduces to "is this player slow?", which is a property of the agent's MCP loop rather than its reasoning.
2. **The game must require a sequence of decisions.** Static one-shot puzzles (the kind benchmarked by OpenCaptchaWorld) are out of scope: we want behavior that unfolds over multiple choices so we can compare *trajectories of decisions*, not single answers.
3. **It must be a viable CAPTCHA.** Short, easy for a human to complete, and deployable inside a normal web flow.

For the scope of this project we use **Claude Code** as the only LLM agent. This lets us reuse its existing agent loop (planning, tool use, screenshots, retries) and gives us a single well-defined token-and-tool budget per session.

## The Two Type of Games We Propose

We proposed two families of games, one for each checkpoint:

- **H1 — Motor control (Trace-the-Tunnel).** A start point, an end point, and a curved tunnel between them on a canvas. The player must drag from start to end without the cursor leaving the tunnel. Picked because human pointer behavior is well studied in HCI, while agent pointer behavior is not.
- **H2 — Decision making under uncertainty (Hover-to-Find).** A grid in which one tile is the target. Each wrong click reveals partial information about where the target is. Picked because humans show *strategy diversity* on this kind of game (some players take small conservative steps; some click broadly first to get a global picture, then zoom in), so the question becomes whether agents reproduce that distribution of strategies.

H1 is implemented and the data has been collected for this checkpoint. H2 will be implemented before checkpoint 2.

---

## Trace-the-Tunnel (this checkpoint)

### Game design

The game renders a curved tunnel on a 600×350 canvas, with a green start dot on the left and a red end dot on the right. The player presses on the green dot and drags along the tunnel to the red dot. If the cursor leaves the tunnel at any point, the run is marked a failure and the same tunnel auto-resets. There is a fixed pool of 10 tunnels; both humans and agents play the same pool.

### Trajectory recording

We log the player's input as an **event-driven** stream. The canvas registers a native `mousemove` listener, and every browser-dispatched event triggers a record of `{x, y, timestamp, event_type, inside_tunnel}`, with the timestamp from `performance.now()`. Mouse-down and mouse-up events are recorded the same way. Because there is no `setInterval` / `requestAnimationFrame` driver and no throttling in our code, inter-event intervals reflect whatever the browser, OS, and input device produce, and a stationary cursor produces no events at all.

### Agent workflow

We compared a single agent stack against humans on the same 10-tunnel pool: a **Claude Code agent given no access to the game source code**, sandboxed to a small set of Playwright tools, perceiving the tunnel only from the rendered page.

The experiment lives in `trace-the-tunnel-exp/`, which is a separate directory from the game source so the agent's CWD never contains `game.js` or any other implementation file. `run.sh` invokes:

```
claude -p "$(cat prompts/visual.txt)" \
  --allowedTools "browser_navigate,browser_take_screenshot,browser_snapshot,browser_click,browser_evaluate"
```

The allowlist is deliberately narrow: visual perception (screenshot/snapshot), a click primitive, and `browser_evaluate` for synthesizing mouse events on a curved path. Notably absent: `Read`, `Bash`, `Grep`, `Glob`, `Edit`, `Write` — so the agent cannot escape to the filesystem and read `game.js` to recover the tunnel geometry directly. It has to look at the canvas.

We ran the agent in **multiple individual sessions with clean context**, both to collect more trajectories and to see whether different sessions would converge on different strategies.

### What the agent actually did

Across rounds, all sessions converged on the same approach: the agent inspected the page, analyzed the canvas pixels to find the tunnel (typically by clustering pixel colors), and then wrote a JS routine (which it consistently called a "solver") that computes a path through the tunnel and dispatches the resulting mouse events via `browser_evaluate`. A representative copy of one such solver is checked in at `trace-the-tunnel-exp/solver.js`.

We also had to ask the agent to introduce variation between runs directly so that successive trajectories are not identical. Without this instruction, the agent tended to emit a deterministic curve, which would have made the population trivially distinguishable for reasons we did not want to study.

Successful trajectories are saved by the game server under `data/visual/`. (The server uses `--experiment visual` to force the output directory regardless of what the agent claims as its source label.)

---

## Hover-to-Find (next checkpoint)

For the next checkpoint we will implement a grid-based decision-under-uncertainty game. The grid contains one hidden target tile. On each click of a wrong tile, the game reveals a partial hint about where the target lies (the exact hint format is still TBD — e.g., "warmer/colder," distance band, or a row/column constraint).

We picked this game because we hypothesize that humans don't share a single optimal strategy here. Some players take small conservative steps from one neighborhood; others click broadly across the grid first to get a global picture before narrowing in. So the comparison we will run isn't "does the agent's path look like the human path?" but rather "does the distribution of game states the agent visits look like the distribution humans visit?"

**Planned evaluation.** The unit of evaluation is the per-step tuple `(revealed_state_so_far, cell_chosen, hint_returned)`. Aggregated across sessions, this gives an empirical distribution over visited states per population. We plan to evaluate at two levels. First, a *distributional* comparison: estimate the human and agent state-visit distributions and compute a divergence (e.g. Jensen-Shannon) between them; for a per-session score, compute the log-likelihood of a candidate session's action sequence under the human distribution and use it as a discriminator. Second, an *interpretable* comparison: bucket each session into a coarse strategy class (e.g. "local-conservative," "broad-then-narrow," "row-or-column scan," "near-random") using a small classifier on the action sequence, and compare the strategy-frequency histograms across populations — a concrete mismatch like *"humans go broad-then-narrow 30%, agents 70%"* is easier to communicate than a divergence number, and it's the type of finding we expect to be most actionable for game design. Consistent with constraint 1, **timing features are deliberately excluded** from the H2 eval — only the *content* of the decision sequence is compared.

