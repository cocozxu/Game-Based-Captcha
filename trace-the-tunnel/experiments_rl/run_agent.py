"""
RL agent runner using Playwright CDP-level mouse dispatch.

Key mechanism advantage over all prior agents:
  - setTimeout (JS replay): each event delayed by macrotask overhead → dt_std inflated
  - This runner: Python asyncio controls timing, events dispatched via
    Playwright's CDP Input.dispatchMouseEvent which feeds the browser's
    real input pipeline → timestamps recorded by game.js via performance.now()
    are set at actual event-receipt time, matching OS input loop behaviour

Usage (server must already be running with --experiment rl_agent):
  cd trace-the-tunnel
  source .venv/bin/activate
  python experiments_rl/run_agent.py [--policy ppo_policy_best.pt] [--rounds 3]
"""

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
import uuid

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from policy import TunnelPolicy, load_policy
from tunnel_env import (
    TunnelEnv, generate_tunnel, is_inside_tunnel,
    TUNNEL_WIDTH, CANVAS_W, CANVAS_H, DOT_RADIUS,
    DT_MEAN_MS, DT_STD_MS,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "tunnel_config.json")
BASE_URL = "http://localhost:5050"

# Timing model (matches human dt distribution)
DT_MIN_MS = 5.5    # never faster than this
DT_MAX_MS = 20.0   # cap individual delays (human p99 ~12ms; allow some wiggle)


# ---------------------------------------------------------------------------
# Human-like timing sampler
# ---------------------------------------------------------------------------

def sample_dt(rng: random.Random) -> float:
    """
    Sample an inter-event delay in milliseconds.

    Models the human dt distribution:
      median ≈ 8.3 ms, within-trace std ≈ 0.9 ms, p5=7.6 p95=9.4
    We use a folded-normal with occasional larger gaps (momentary hesitation).
    """
    # Base Gaussian component
    dt = rng.gauss(DT_MEAN_MS, DT_STD_MS)

    # Rare hesitations (~5% of steps): add extra 1-8ms pause
    if rng.random() < 0.05:
        dt += rng.uniform(1.0, 8.0)

    return max(DT_MIN_MS, min(DT_MAX_MS, dt))


# ---------------------------------------------------------------------------
# Trajectory generation from policy
# ---------------------------------------------------------------------------

def generate_trajectory(
    env: TunnelEnv,
    policy: TunnelPolicy,
    rng: random.Random,
    max_steps: int = 1500,
    jitter_px: float = 0.4,
) -> list[dict]:
    """
    Run the trained policy in simulation and produce a timed event list.

    Returns a list of dicts: {x, y, dt_ms, event_type}
    where dt_ms is the delay *before* this event.
    """
    obs, _ = env.reset()
    events = []

    # mousedown at start
    events.append({
        "x": env._x,
        "y": env._y,
        "dt_ms": 0.0,
        "event_type": "mousedown",
    })

    for step in range(max_steps):
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = policy.get_action(obs_t, deterministic=False)
        action_np = action.squeeze(0).numpy()

        obs, reward, done, truncated, info = env.step(action_np)

        # Add sub-pixel jitter: humans are never pixel-perfect
        jx = rng.gauss(0.0, jitter_px)
        jy = rng.gauss(0.0, jitter_px)
        x_emit = float(np.clip(env._x + jx, 0, CANVAS_W))
        y_emit = float(np.clip(env._y + jy, 0, CANVAS_H))

        # Verify the jittered position is still inside (clip back if not)
        if not is_inside_tunnel(x_emit, y_emit, env.centerline):
            x_emit, y_emit = env._x, env._y

        dt_ms = sample_dt(rng)
        events.append({
            "x": x_emit,
            "y": y_emit,
            "dt_ms": dt_ms,
            "event_type": "mousemove",
        })

        if done or truncated:
            break

    # mouseup at last position
    events.append({
        "x": events[-1]["x"],
        "y": events[-1]["y"],
        "dt_ms": sample_dt(rng),
        "event_type": "mouseup",
    })

    return events, info


# ---------------------------------------------------------------------------
# Playwright dispatch
# ---------------------------------------------------------------------------

async def dispatch_trajectory(page, events: list[dict], canvas_rect: dict):
    """
    Dispatch mouse events via Playwright CDP.

    Playwright's page.mouse API routes through CDP Input.dispatchMouseEvent,
    which is processed by the browser's real input pipeline. Timestamps are
    set by performance.now() at the moment the browser fires the event
    handler — the same as OS hardware input. This is fundamentally different
    from JS setTimeout-based dispatch.
    """
    # Translate canvas-relative coords to page coords
    def to_page(x, y):
        px = canvas_rect["x"] + x * (canvas_rect["width"] / CANVAS_W)
        py = canvas_rect["y"] + y * (canvas_rect["height"] / CANVAS_H)
        return px, py

    # Move mouse to start without pressing
    first = events[0]
    px, py = to_page(first["x"], first["y"])
    await page.mouse.move(px, py)
    await asyncio.sleep(0.05)  # brief pause before pressing

    # mousedown
    await page.mouse.down()
    await asyncio.sleep(0.01)

    # mousemove events with precise timing
    for evt in events[1:]:
        if evt["event_type"] == "mouseup":
            break
        dt_s = evt["dt_ms"] / 1000.0
        await asyncio.sleep(dt_s)
        px, py = to_page(evt["x"], evt["y"])
        await page.mouse.move(px, py)

        # Check game state every few steps
        state = await page.evaluate("window.__tunnelGame.getState()")
        if state in ("done_success", "done_fail"):
            break

    await page.mouse.up()
    await asyncio.sleep(0.05)

    return await page.evaluate("window.__tunnelGame.getState()")


async def save_session(page, experiment_name: str):
    """POST the current session to the server via the game's save endpoint."""
    result = await page.evaluate(f"""
        async () => {{
            const data = window.__tunnelGame.getSessionData();
            data.source = '{experiment_name}';
            const resp = await fetch('/api/save_trajectory', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify(data),
            }});
            return await resp.json();
        }}
    """)
    return result


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

async def run(policy_path: str, n_rounds: int, experiment_name: str, headless: bool):
    from playwright.async_api import async_playwright

    with open(CONFIG_PATH) as f:
        config = json.load(f)
    tunnel_ids = [t["tunnel_id"] for t in config["tunnels"]]

    policy = load_policy(policy_path)
    policy.eval()
    print(f"Loaded policy: {policy_path}")

    rng = random.Random()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        await page.goto(BASE_URL)
        await page.wait_for_load_state("networkidle")

        # Wait for tunnel pool to load
        await asyncio.sleep(0.5)

        # Verify server is in correct experiment mode
        mode = await page.evaluate("async () => (await fetch('/api/mode')).json()")
        if mode.get("experiment") != experiment_name:
            print(f"ERROR: server is in mode '{mode.get('experiment')}', expected '{experiment_name}'")
            print(f"  Start server with: python server.py --experiment {experiment_name}")
            await browser.close()
            return

        stats = {"success": 0, "fail": 0, "total": 0}

        for round_num in range(1, n_rounds + 1):
            print(f"\n=== Round {round_num}/{n_rounds} ===")

            for tid in tunnel_ids:
                # Load tunnel
                loaded = await page.evaluate(f"window.__tunnelGame.loadTunnel({tid})")
                if not loaded:
                    print(f"  tunnel {tid}: failed to load — skipping")
                    continue
                await asyncio.sleep(0.15)

                canvas_rect = await page.evaluate("""
                    () => {
                        const r = document.querySelector('canvas').getBoundingClientRect();
                        return {x: r.left, y: r.top, width: r.width, height: r.height};
                    }
                """)

                # Reconstruct Python env for this tunnel
                env = TunnelEnv(tunnel_id=tid, seed=rng.randint(0, 2**31))

                # Try up to 3 times per tunnel
                for attempt in range(1, 4):
                    # Reload tunnel (resets game state)
                    await page.evaluate(f"window.__tunnelGame.loadTunnel({tid})")
                    await asyncio.sleep(0.1)

                    # Generate trajectory from policy
                    env = TunnelEnv(tunnel_id=tid, seed=rng.randint(0, 2**31))
                    events, info = generate_trajectory(env, policy, rng)

                    n_moves = sum(1 for e in events if e["event_type"] == "mousemove")
                    sim_reason = info.get("reason", "unknown")
                    print(f"  tunnel {tid} attempt {attempt}: sim={sim_reason} ({n_moves} moves) ", end="", flush=True)

                    if sim_reason != "success":
                        print("→ skip (policy failed in sim)")
                        continue

                    # Dispatch in browser
                    final_state = await dispatch_trajectory(page, events, canvas_rect)
                    print(f"→ browser={final_state}", end="")

                    if final_state == "done_success":
                        save_result = await save_session(page, experiment_name)
                        print(f"  saved: {save_result}")
                        stats["success"] += 1
                        stats["total"] += 1
                        break
                    else:
                        print()
                        stats["fail"] += 1
                        stats["total"] += 1

                await asyncio.sleep(0.2)

        print(f"\n=== Done. Success: {stats['success']}/{stats['total']} "
              f"({100*stats['success']/max(stats['total'],1):.0f}%) ===")
        await browser.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="ppo_policy_best.pt",
                        help="Policy filename inside experiments_rl/models/")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--experiment", default="rl_agent",
                        help="Must match --experiment on the running server")
    parser.add_argument("--headless", action="store_true",
                        help="Run browser headlessly")
    args = parser.parse_args()

    policy_path = os.path.join(MODEL_DIR, args.policy)
    if not os.path.exists(policy_path):
        print(f"Policy not found: {policy_path}")
        print("Run bc_train.py and ppo_train.py first.")
        sys.exit(1)

    asyncio.run(run(
        policy_path=policy_path,
        n_rounds=args.rounds,
        experiment_name=args.experiment,
        headless=args.headless,
    ))
