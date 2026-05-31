"""
run_solver.py — runs an agent-written solver through the forge against
the test tunnels of one split.

Workflow per (split_id, model):
  1. Load solvers/{split_id}/{model}.py dynamically.
  2. For each test tunnel in splits.yaml[{split_id}].test:
       - Build tunnel_spec from any on-disk human trace for that tunnel.
       - For seed in 0..traces_per_test_tunnel-1:
           - call solver.generate(tunnel_spec, seed) inside a timeout
           - validate the returned events
           - dispatch via lib.forge → captcha gate
       - Record per-attempt outcome: success / solver_error / malformed /
         captcha_reject.
  3. Write manifest_<ts>.json into data/gen_{model}_{split_id}/.

Server must be running as:
  python server.py --experiment gen_{model}_{split_id} --expose-debug

The script verifies the running server's --experiment matches before
dispatching anything.
"""

import argparse
import asyncio
import glob
import importlib.util
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

import requests
import yaml
from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)

from lib.forge import (
    install_forge,
    dispatch_trace,
    get_canvas_geometry,
    load_tunnel,
    wait_for_state,
)

DATA_DIR = os.path.join(REPO, "data")
HUMAN_DIR = os.path.join(DATA_DIR, "human")
TUNNEL_IDS = list(range(10))

VALID_EVENT_TYPES = {"mousedown", "mousemove", "mouseup"}


# ---------------------------------------------------------------------------
# Solver loading
# ---------------------------------------------------------------------------

def load_solver(path):
    spec = importlib.util.spec_from_file_location("agent_solver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "generate"):
        raise RuntimeError(f"{path} does not define generate(tunnel_spec, seed)")
    return module


# ---------------------------------------------------------------------------
# Tunnel-spec construction (geometry only — same shape as in the prompt)
# ---------------------------------------------------------------------------

def load_tunnel_specs():
    """Pick one human trace per tunnel to source geometry from."""
    specs = {}
    for path in glob.glob(os.path.join(HUMAN_DIR, "*.json")):
        try:
            with open(path) as f:
                t = json.load(f)
        except Exception:
            continue
        tid = t.get("tunnel_id")
        if tid in TUNNEL_IDS and tid not in specs:
            specs[tid] = {
                "tunnel_id": t["tunnel_id"],
                "tunnel_seed": t["tunnel_seed"],
                "control_points": t["control_points"],
                "tunnel_width": t["tunnel_width"],
                "canvas_size": t["canvas_size"],
                "viewport": t["viewport"],
            }
        if len(specs) == len(TUNNEL_IDS):
            break
    missing = [t for t in TUNNEL_IDS if t not in specs]
    if missing:
        raise RuntimeError(f"no human trace on disk for tunnels {missing}")
    return specs


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_events(events, tunnel_spec):
    """Return (ok, reason). Reason is a short string when ok is False."""
    if not isinstance(events, list):
        return False, f"return type {type(events).__name__}, expected list"
    if len(events) < 2:
        return False, f"too few events ({len(events)})"
    for i, e in enumerate(events):
        if not isinstance(e, dict):
            return False, f"event {i} is not a dict ({type(e).__name__})"
        for k in ("x", "y", "timestamp", "event_type"):
            if k not in e:
                return False, f"event {i} missing field {k!r}"
        et = e["event_type"]
        if et not in VALID_EVENT_TYPES:
            return False, f"event {i} has invalid event_type {et!r}"
        for k in ("x", "y", "timestamp"):
            if not isinstance(e[k], (int, float)):
                return False, f"event {i}.{k} not numeric ({type(e[k]).__name__})"

    # Exactly one mousedown (first), exactly one mouseup (last).
    if events[0]["event_type"] != "mousedown":
        return False, f"first event is {events[0]['event_type']!r}, expected mousedown"
    if events[-1]["event_type"] != "mouseup":
        return False, f"last event is {events[-1]['event_type']!r}, expected mouseup"
    n_down = sum(1 for e in events if e["event_type"] == "mousedown")
    n_up = sum(1 for e in events if e["event_type"] == "mouseup")
    if n_down != 1:
        return False, f"{n_down} mousedown events, expected 1"
    if n_up != 1:
        return False, f"{n_up} mouseup events, expected 1"

    # Strictly monotonic timestamps.
    for i in range(1, len(events)):
        if events[i]["timestamp"] < events[i - 1]["timestamp"]:
            return False, f"timestamp regression at event {i}: {events[i-1]['timestamp']} -> {events[i]['timestamp']}"

    # Coordinates within canvas bounds.
    cs = tunnel_spec["canvas_size"]
    w = cs["width"] if isinstance(cs, dict) else cs[0]
    h = cs["height"] if isinstance(cs, dict) else cs[1]
    for i, e in enumerate(events):
        if not (-1 <= e["x"] <= w + 1) or not (-1 <= e["y"] <= h + 1):
            return False, f"event {i} ({e['x']}, {e['y']}) outside canvas {w}x{h}"

    return True, None


def coerce_events(events):
    """Normalize agent output to the dispatch format. Fills inside_tunnel
    if missing (defaults True; the captcha computes its own boundary check
    anyway). Does not modify validation-relevant fields."""
    out = []
    for e in events:
        out.append({
            "x": float(e["x"]),
            "y": float(e["y"]),
            "timestamp": float(e["timestamp"]),
            "event_type": e["event_type"],
            "inside_tunnel": bool(e.get("inside_tunnel", True)),
        })
    return out


# ---------------------------------------------------------------------------
# Per-attempt runner
# ---------------------------------------------------------------------------

async def run_attempt(page, cdp, tunnel_spec, seed, solver, executor, out_dir, timeout_s):
    """One generate() → dispatch → wait. Returns an attempt-record dict."""
    rec = {
        "tunnel_id": tunnel_spec["tunnel_id"],
        "seed": seed,
        "stage": None,
        "reason": None,
        "saved": None,
        "n_dispatched": None,
        "n_events": None,
        "duration_s": None,
    }
    t0 = time.monotonic()

    # ---- stage: solver invocation ----
    try:
        fut = executor.submit(solver.generate, tunnel_spec, seed)
        events = fut.result(timeout=timeout_s)
    except FutTimeout:
        rec["stage"] = "solver_error"
        rec["reason"] = f"timeout > {timeout_s}s"
        rec["duration_s"] = round(time.monotonic() - t0, 3)
        return rec
    except Exception as e:
        rec["stage"] = "solver_error"
        rec["reason"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
        rec["duration_s"] = round(time.monotonic() - t0, 3)
        return rec

    rec["n_events"] = len(events) if isinstance(events, list) else None

    # ---- stage: output validation ----
    ok, reason = validate_events(events, tunnel_spec)
    if not ok:
        rec["stage"] = "malformed"
        rec["reason"] = reason
        rec["duration_s"] = round(time.monotonic() - t0, 3)
        return rec
    events = coerce_events(events)

    # ---- stage: captcha dispatch ----
    try:
        await load_tunnel(page, tunnel_spec["tunnel_id"])
        geom = await get_canvas_geometry(page)
        before = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
        n = await dispatch_trace(cdp, page, events, geom)
        rec["n_dispatched"] = n
        await asyncio.sleep(0.1)
        state = await wait_for_state(page, ("done_success", "done_fail"), timeout_ms=2000)
    except Exception as e:
        rec["stage"] = "captcha_reject"
        rec["reason"] = f"dispatch exception: {type(e).__name__}: {e}"
        rec["duration_s"] = round(time.monotonic() - t0, 3)
        return rec

    if state != "done_success":
        rec["stage"] = "captcha_reject"
        rec["reason"] = f"end state {state!r}"
        rec["duration_s"] = round(time.monotonic() - t0, 3)
        await page.evaluate("() => { window.__forge.consumed = 0; }")
        return rec

    # ---- stage: success — wait for save file ----
    saved_name = None
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        await asyncio.sleep(0.1)
        if os.path.isdir(out_dir):
            new = set(os.listdir(out_dir)) - before
            new = [n for n in new if not n.startswith("manifest_")]
            if new:
                saved_name = new[0]
                break

    await page.evaluate("() => { window.__forge.consumed = 0; }")

    if not saved_name:
        rec["stage"] = "captcha_reject"
        rec["reason"] = "done_success but no file appeared"
    else:
        rec["stage"] = "success"
        rec["saved"] = saved_name
    rec["duration_s"] = round(time.monotonic() - t0, 3)
    return rec


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run(args):
    with open(args.splits) as f:
        cfg = yaml.safe_load(f)
    splits_by_id = {s["id"]: s for s in cfg["splits"]}
    if args.split not in splits_by_id:
        raise SystemExit(f"split {args.split!r} not in {args.splits}; got {list(splits_by_id)}")
    split = splits_by_id[args.split]
    traces_per = cfg["generation"]["traces_per_test_tunnel"]

    solver_path = os.path.join(HERE, "solvers", split["id"], f"{args.model}.py")
    if not os.path.exists(solver_path):
        raise SystemExit(f"missing solver: {solver_path}\nPaste agent output into that path first.")
    solver = load_solver(solver_path)
    print(f"Loaded solver: {solver_path}")

    expected_experiment = f"gen_{args.model}_{split['id']}"
    info = requests.get(f"{args.server}/api/mode", timeout=5).json()
    if info.get("experiment") != expected_experiment:
        raise SystemExit(
            f"server is running as experiment {info.get('experiment')!r}; "
            f"start it as: python server.py --experiment {expected_experiment} --expose-debug"
        )
    if not info.get("expose_debug"):
        raise SystemExit("server must be started with --expose-debug (window.__tunnelGame)")

    out_dir = os.path.join(DATA_DIR, expected_experiment)
    os.makedirs(out_dir, exist_ok=True)
    specs = load_tunnel_specs()

    attempts = []
    started = time.time()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=args.headless)
        ctx = await browser.new_context(viewport={"width": 1024, "height": 768})
        await install_forge(ctx)
        page = await ctx.new_page()
        await page.goto(args.server + "/", wait_until="load")
        await page.wait_for_function("() => !!window.__tunnelGame && !!window.__forge", timeout=10000)
        cdp = await ctx.new_cdp_session(page)

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            for tid in split["test"]:
                tunnel_spec = specs[tid]
                consecutive_solver_errors = 0
                for seed in range(traces_per):
                    print(f"[tunnel {tid} seed {seed}] ...", flush=True, end=" ")
                    rec = await run_attempt(
                        page, cdp, tunnel_spec, seed, solver, executor, out_dir, args.timeout
                    )
                    attempts.append(rec)
                    print(f"{rec['stage']}" + (f" ({rec['reason'][:80]})" if rec.get("reason") else ""), flush=True)
                    if rec["stage"] == "solver_error":
                        consecutive_solver_errors += 1
                    else:
                        consecutive_solver_errors = 0
                    if consecutive_solver_errors >= 3:
                        print(f"  ABORTING tunnel {tid}: {consecutive_solver_errors} consecutive solver_errors")
                        break
        finally:
            executor.shutdown(wait=False)
            await browser.close()

    counts = {"success": 0, "solver_error": 0, "malformed": 0, "captcha_reject": 0}
    for r in attempts:
        counts[r["stage"]] = counts.get(r["stage"], 0) + 1

    manifest = {
        "experiment": expected_experiment,
        "split_id": split["id"],
        "model": args.model,
        "demo": split["demo"],
        "test": split["test"],
        "traces_per_test_tunnel": traces_per,
        "started": started,
        "finished": time.time(),
        "counts": counts,
        "n_attempts": len(attempts),
        "attempts": attempts,
    }
    manifest_path = os.path.join(out_dir, f"manifest_{int(started)}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {counts}")
    print(f"Manifest: {manifest_path}")
    return 0 if counts.get("success", 0) > 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="split id from splits.yaml (e.g. split_a)")
    ap.add_argument("--model", required=True, help="model label (e.g. opus, sonnet, haiku) — must match solvers/<split>/<model>.py")
    ap.add_argument("--server", default="http://localhost:5050")
    ap.add_argument("--splits", default=os.path.join(HERE, "splits.yaml"))
    ap.add_argument("--timeout", type=float, default=30.0, help="seconds before a single generate() call is killed")
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--headed", dest="headless", action="store_false")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
