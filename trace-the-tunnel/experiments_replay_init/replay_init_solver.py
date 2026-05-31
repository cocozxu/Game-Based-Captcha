"""
replay_init_solver.py — Patch A: forge what the captcha *reads*, not what
the OS *delivers*.

The replay_cdp and replay_hid experiments tried to drive perfect
`(x, y, t)` through the OS/browser pipeline so the captcha's recorded
values would match the source. Both failed at AUC 0.90+ because every
hop in that pipeline (CGEvent quantization, renderer scheduling, integer
clientX/Y) is lossy.

This solver gives up on the pipeline and forges the read side. Before
the captcha's `game.js` loads, we inject an init script that:

  - replaces `MouseEvent.prototype.clientX` / `clientY` getters to return
    `source.x + rect.left` / `source.y + rect.top`
  - replaces `performance.now()` to return `source.timestamp`
  - replaces `Date.now()` derived from `performance.timeOrigin + now()`

Each forged read consumes from a queue we populate before dispatching.
The CAPTCHA's handler sees source values regardless of what the OS or
the renderer scheduler actually delivered. Recorded JSON should be
byte-identical to the source human's JSON on (x, y, t).

Dispatch channel is still CDP `Input.dispatchMouseEvent` — we don't
need precision delivery anymore since the read values are forged, but
events still need to *arrive* so the handler fires.

The forge JS + dispatch loop live in `lib/forge.py` so other experiments
(experiments_generalize, etc.) can reuse the same channel.

The server must be running with:
  --experiment replay_init --expose-debug \\
  --allowed-sessions experiments_replay/allowed_sessions_v1b.json
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time

import requests
from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
DATA_DIR = os.path.join(REPO, "data")
OUT_DIR = os.path.join(DATA_DIR, "replay_init")

from lib.forge import (
    INIT_SCRIPT,
    install_forge,
    dispatch_trace,
    get_canvas_geometry,
    load_tunnel,
    wait_for_state,
    slice_down_to_up,
)


# ---------------------------------------------------------------------------
# Source-trace helpers
# ---------------------------------------------------------------------------

def dt_stats(events):
    ts = [e["timestamp"] for e in events if e["event_type"] == "mousemove"]
    if len(ts) < 2:
        return None, None, len(ts)
    diffs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    return statistics.fmean(diffs), statistics.pstdev(diffs), len(ts)


def find_source_in_bank(bank, session_id):
    for src in bank:
        if src["session_id"] == session_id:
            return src
    return None


def fetch_bank(server, tid, timeout=10):
    r = requests.get(f"{server}/api/human_bank/{tid}", timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# One attempt
# ---------------------------------------------------------------------------

async def run_one(page, cdp, tid, source_trace):
    await load_tunnel(page, tid)
    geom = await get_canvas_geometry(page)
    n = await dispatch_trace(cdp, page, source_trace["events"], geom)
    await asyncio.sleep(0.1)
    st = await wait_for_state(page, ("done_success", "done_fail"), timeout_ms=1500)
    return {"state": st, "n_dispatched": n}


# ---------------------------------------------------------------------------
# Collect mode
# ---------------------------------------------------------------------------

async def cmd_collect(args):
    rng = random.Random(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=args.headless)
        ctx = await browser.new_context(viewport={"width": 1024, "height": 768})
        await install_forge(ctx)
        page = await ctx.new_page()
        await page.goto(args.server + "/", wait_until="load")
        await page.wait_for_function("() => !!window.__tunnelGame && !!window.__forge", timeout=10000)
        cdp = await ctx.new_cdp_session(page)

        banks = {}
        for tid in range(10):
            banks[tid] = fetch_bank(args.server, tid)
            print(f"[bank] tunnel {tid}: {len(banks[tid])} sources")

        results = []
        manifest_started = time.time()
        for tid in range(10):
            bank = banks[tid]
            if not bank:
                print(f"[tunnel {tid}] empty bank")
                continue
            slot = 0
            failures = 0
            while slot < args.per_tunnel and failures < 4 * args.per_tunnel:
                choice = rng.choice(bank)
                print(f"[tunnel {tid} slot {slot}] source={choice['session_id'][:8]} events={len(choice['events'])}", flush=True)
                before = set(os.listdir(OUT_DIR))
                try:
                    r = await run_one(page, cdp, tid, choice)
                except Exception as e:
                    print(f"  -> exception: {e}", flush=True)
                    failures += 1
                    continue
                if r["state"] == "done_success":
                    saved_name = None
                    deadline = time.monotonic() + 1.5
                    while time.monotonic() < deadline:
                        await asyncio.sleep(0.1)
                        new = set(os.listdir(OUT_DIR)) - before
                        new = [n for n in new if not n.startswith("manifest_")]
                        if new:
                            saved_name = new[0]
                            break
                    if saved_name:
                        consumed = await page.evaluate("() => window.__forge.consumed")
                        print(f"  -> SUCCESS saved={saved_name} forge_consumed={consumed}/{r['n_dispatched']}", flush=True)
                        results.append({
                            "tunnel_id": tid,
                            "slot": slot,
                            "source_session": choice["session_id"],
                            "saved": saved_name,
                            "n_dispatched": r["n_dispatched"],
                            "forge_consumed": consumed,
                        })
                        slot += 1
                    else:
                        print(f"  -> done_success but no file appeared", flush=True)
                        failures += 1
                else:
                    print(f"  -> state={r['state']} n_dispatched={r['n_dispatched']}", flush=True)
                    failures += 1
                await page.evaluate("() => { window.__forge.consumed = 0; }")

        n_saved = sum(1 for r in results if r.get("saved"))
        manifest = {
            "experiment": "replay_init",
            "started": manifest_started,
            "finished": time.time(),
            "per_tunnel": args.per_tunnel,
            "seed": args.seed,
            "n_saved": n_saved,
            "results": results,
        }
        manifest_path = os.path.join(OUT_DIR, f"manifest_{int(manifest_started)}.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nSaved {n_saved} successful traces. Manifest: {manifest_path}")

        await browser.close()


# ---------------------------------------------------------------------------
# Validate mode — verify byte-identity against source
# ---------------------------------------------------------------------------

async def cmd_validate(args):
    print("=" * 60)
    print("Validation checks (replay_init / forge)")
    print("=" * 60)

    manifests = sorted([f for f in os.listdir(OUT_DIR) if f.startswith("manifest_") and f.endswith(".json")])
    if not manifests:
        print("FAIL: no manifest")
        return 1
    saved_entries = []
    for m in manifests:
        with open(os.path.join(OUT_DIR, m)) as f:
            saved_entries.extend([r for r in json.load(f).get("results", []) if r.get("saved")])
    if not saved_entries:
        print("FAIL: no saved entries")
        return 1
    print(f"\n{len(saved_entries)} saved entries across {len(manifests)} manifest(s)")

    banks = {}
    for tid in range(10):
        banks[tid] = fetch_bank(args.server, tid)

    overall_ok = True

    print("\n--- Check 1: aggregate dt drift (expect ≈0) ---")
    dms, dss = [], []
    for r in saved_entries:
        sp = os.path.join(OUT_DIR, r["saved"])
        if not os.path.exists(sp):
            continue
        s = json.load(open(sp))
        src = find_source_in_bank(banks.get(s["tunnel_id"], []), r["source_session"])
        if src is None:
            continue
        sm, ss, _ = dt_stats(s["events"])
        cm, cs, _ = dt_stats(src["events"])
        if sm is None or cm is None:
            continue
        dms.append(abs(sm - cm))
        dss.append(abs(ss - cs))
    if dms:
        print(f"  n compared = {len(dms)}")
        print(f"  |Δdt_mean| median={statistics.median(dms):.6f} ms max={max(dms):.6f} ms")
        print(f"  |Δdt_std|  median={statistics.median(dss):.6f} ms max={max(dss):.6f} ms")
        if max(dms) < 0.01 and max(dss) < 0.01:
            print("  PASS (byte-identical timing)")
        elif statistics.median(dms) <= 0.05 and statistics.median(dss) <= 0.1:
            print("  PASS (sub-tolerance, see notes)")
        else:
            print("  FAIL")
            overall_ok = False

    print("\n--- Check 2: aggregate (x, y) drift (expect ≈0) ---")
    all_dx, all_dy = [], []
    pairs = 0
    for r in saved_entries:
        sp = os.path.join(OUT_DIR, r["saved"])
        if not os.path.exists(sp):
            continue
        s = json.load(open(sp))
        src = find_source_in_bank(banks.get(s["tunnel_id"], []), r["source_session"])
        if src is None:
            continue
        sxy = [(e["x"], e["y"]) for e in s["events"] if e["event_type"] == "mousemove"]
        cxy = [(e["x"], e["y"]) for e in src["events"] if e["event_type"] == "mousemove"]
        n = min(len(sxy), len(cxy))
        for i in range(n):
            all_dx.append(abs(sxy[i][0] - cxy[i][0]))
            all_dy.append(abs(sxy[i][1] - cxy[i][1]))
        pairs += 1
    if all_dx:
        print(f"  compared {pairs} traces, {len(all_dx)} mousemoves")
        print(f"  |Δx| mean={statistics.fmean(all_dx):.6f} px max={max(all_dx):.6f} px")
        print(f"  |Δy| mean={statistics.fmean(all_dy):.6f} px max={max(all_dy):.6f} px")
        if max(all_dx) < 1e-3 and max(all_dy) < 1e-3:
            print("  PASS (byte-identical coordinates)")
        elif statistics.fmean(all_dx) < 1.5 and statistics.fmean(all_dy) < 1.5:
            print("  PASS (sub-tolerance)")
        else:
            print("  FAIL")
            overall_ok = False

    print("\n--- Check 3: byte-identical events on sampled trace ---")
    pick = saved_entries[len(saved_entries) // 2]
    s = json.load(open(os.path.join(OUT_DIR, pick["saved"])))
    src = find_source_in_bank(banks.get(s["tunnel_id"], []), pick["source_session"])
    if src is None:
        print("  SKIP: source not found")
    else:
        s_events = slice_down_to_up(s["events"])
        c_events = slice_down_to_up(src["events"])
        print(f"  saved sliced events: {len(s_events)}  source sliced: {len(c_events)}")
        n = min(len(s_events), len(c_events))
        mismatches = []
        for i in range(n):
            for k in ("x", "y", "timestamp", "event_type"):
                if s_events[i].get(k) != c_events[i].get(k):
                    mismatches.append((i, k, s_events[i].get(k), c_events[i].get(k)))
                    if len(mismatches) >= 5:
                        break
            if len(mismatches) >= 5:
                break
        if not mismatches and len(s_events) == len(c_events):
            print("  PASS (every (x, y, t, type) field equal)")
        else:
            print(f"  {'FAIL' if mismatches else 'PASS (lengths differ; values match where compared)'}")
            for m in mismatches:
                print(f"    [{m[0]}] {m[1]}: saved={m[2]!r}  source={m[3]!r}")
            if mismatches:
                overall_ok = False

    print("\n" + "=" * 60)
    print("RESULT:", "PASS" if overall_ok else "FAIL")
    print("=" * 60)
    return 0 if overall_ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:5050")
    ap.add_argument("--per-tunnel", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--headed", dest="headless", action="store_false")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    if args.validate_only:
        sys.exit(asyncio.run(cmd_validate(args)))
    else:
        asyncio.run(cmd_collect(args))


if __name__ == "__main__":
    main()
