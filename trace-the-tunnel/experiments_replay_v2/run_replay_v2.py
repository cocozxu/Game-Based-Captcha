"""replay_v2 harness — same per-tunnel bank-replay as v1b, but dispatched
through Playwright's CDP-level mouse API (page.mouse.{move,down,up}) instead
of in-page MouseEvent + dispatchEvent.

CDP routes mouse events through Chrome DevTools Protocol's
Input.dispatchMouseEvent, which originates in the browser process. Resulting
events have isTrusted=true and bypass the in-page scheduler overhead
(setTimeout + MouseEvent construction + dispatchEvent) that produced v1b's
dt_mean / dt_std signal.

Hypothesis: if RF AUC drops sharply, v1b's residual signal was the dispatcher
itself, and the captcha's effective defense is channel-bound rather than
behavior-bound.

Same bank (allowed_sessions_v1b.json), same sampling (seeded RNG), same target
tunnels (0..9), same retry policy (<=3 attempts/tunnel), same shape (3 rounds
* 10 tunnels). The ONLY change vs v1b is the dispatch primitive.

Requires: playwright (pip install playwright && playwright install chromium).
Assumes the server is running on localhost:5050 in --experiment replay_v2 mode
with --allowed-sessions experiments_replay/allowed_sessions_v1b.json.
"""

import argparse
import asyncio
import datetime as _dt
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_PATH = REPO_ROOT / "experiments_replay" / "allowed_sessions_v1b.json"
DEFAULT_SERVER = "http://localhost:5050"


def sha12(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def pick_source(bank: list, seed: int) -> dict:
    """Deterministically pick one trace from the bank for a given seed.

    Mirrors v1b's solver: a per-(tunnel, slot, attempt) seed indexes the bank
    with `(rng() * len(bank)) | 0`. We use Python's random for portability
    but with the same shape: one float in [0, 1) scaled to len(bank).
    """
    rng = random.Random(seed)
    idx = int(rng.random() * len(bank))
    if idx >= len(bank):
        idx = len(bank) - 1
    return bank[idx]


def derive_seed(base_seed: int, tunnel_id: int, slot: int, attempt: int) -> int:
    """Same arithmetic as replay_solver.js: baseSeed + slot*97 + tid*7919 + attempt*31."""
    return (base_seed + slot * 97 + tunnel_id * 7919 + attempt * 31) & 0x7FFFFFFF


def trim_to_active(events: list) -> list:
    """Slice from first mousedown to last mouseup, matching replay_solver.js."""
    if not events:
        return events
    down_idx = next((i for i, e in enumerate(events) if e.get("event_type") == "mousedown"), -1)
    up_idx = -1
    for i in range(len(events) - 1, -1, -1):
        if events[i].get("event_type") == "mouseup":
            up_idx = i
            break
    start = down_idx if down_idx >= 0 else 0
    end = up_idx if up_idx >= 0 else len(events) - 1
    return events[start:end + 1]


async def fetch_bank(page, tunnel_id: int) -> list:
    """Fetch the bank for a tunnel via page.evaluate -> in-page fetch."""
    js = """
        async (tid) => {
            const r = await fetch('/api/human_bank/' + tid);
            if (!r.ok) throw new Error('bank fetch ' + r.status);
            return await r.json();
        }
    """
    return await page.evaluate(js, tunnel_id)


async def load_tunnel(page, tunnel_id: int, timeout_ms: int = 3000) -> bool:
    """Call window.__tunnelGame.loadTunnel and wait until state=='ready'."""
    await page.evaluate("(tid) => window.__tunnelGame.loadTunnel(tid)", tunnel_id)
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        st = await page.evaluate("() => window.__tunnelGame.getState()")
        if st == "ready":
            return True
        await asyncio.sleep(0.05)
    return False


async def get_canvas_origin(page) -> dict:
    return await page.evaluate(
        """() => {
            const r = document.querySelector('canvas').getBoundingClientRect();
            return { x: r.left, y: r.top };
        }"""
    )


async def get_state(page) -> str:
    return await page.evaluate("() => window.__tunnelGame.getState()")


async def save_via_page(page, source_name: str) -> dict:
    """Send the live session via the page's fetch to keep the same origin.

    The server in --experiment replay_v2 mode will overwrite data['source']
    regardless, but we set it client-side for hygiene + symmetry with v1b.
    """
    js = """
        async (src) => {
            const data = window.__tunnelGame.getSessionData();
            data.source = src;
            const resp = await fetch('/api/save_trajectory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            return await resp.json();
        }
    """
    return await page.evaluate(js, source_name)


async def dispatch_via_cdp(page, events: list, origin: dict) -> None:
    """Replay events through page.mouse.{move,down,up} (CDP-backed)."""
    slice_ = trim_to_active(events)
    if not slice_:
        return

    first = slice_[0]
    # Move to start position first (no button), then press down at the first event.
    await page.mouse.move(origin["x"] + first["x"], origin["y"] + first["y"])
    await page.mouse.down(button="left")

    last_t = first["timestamp"]
    for i in range(1, len(slice_)):
        e = slice_[i]
        dt = max(0.0, (e["timestamp"] - last_t) / 1000.0)
        last_t = e["timestamp"]
        if dt > 0:
            await asyncio.sleep(dt)

        et = e.get("event_type")
        if et == "mousemove":
            await page.mouse.move(origin["x"] + e["x"], origin["y"] + e["y"])
        elif et == "mousedown":
            # Source already has a leading mousedown that we consumed above.
            # Defensive: if another mousedown appears mid-trace, route to move
            # to avoid double-press, matching v1b's "dispatch the event type"
            # behavior loosely (real human traces only have one mousedown).
            await page.mouse.move(origin["x"] + e["x"], origin["y"] + e["y"])
        elif et == "mouseup":
            await page.mouse.move(origin["x"] + e["x"], origin["y"] + e["y"])

        # Early-break if the game already terminated.
        st = await get_state(page)
        if st in ("done_success", "done_fail"):
            break

    # Final release. Position is wherever the last move landed.
    await page.mouse.up(button="left")


async def run_one_attempt(page, tunnel_id: int, bank: list, seed: int) -> dict:
    """One replay attempt: load tunnel, pick source, dispatch, check, save."""
    if not await load_tunnel(page, tunnel_id):
        return {"tunnel_id": tunnel_id, "state": "load_timeout", "save": None}

    source = pick_source(bank, seed)
    origin = await get_canvas_origin(page)
    await dispatch_via_cdp(page, source["events"], origin)
    await asyncio.sleep(0.08)
    state = await get_state(page)
    save_result = None
    if state == "done_success":
        save_result = await save_via_page(page, "replay_v2")
    return {
        "tunnel_id": tunnel_id,
        "state": state,
        "save": save_result,
        "source_session": source.get("session_id"),
        "source_event_count": len(source.get("events", [])),
        "bank_size": len(bank),
        "seed": seed,
    }


async def run_tunnel(page, tunnel_id: int, slot: int, base_seed: int, max_attempts: int = 3) -> dict:
    """Run up to max_attempts on a single (tunnel, slot)."""
    bank = await fetch_bank(page, tunnel_id)
    if not bank:
        return {"tunnel_id": tunnel_id, "slot": slot, "state": "empty_bank", "attempts": []}

    attempts = []
    for attempt in range(max_attempts):
        seed = derive_seed(base_seed, tunnel_id, slot, attempt)
        result = await run_one_attempt(page, tunnel_id, bank, seed)
        result["attempt"] = attempt
        attempts.append(result)
        if result["state"] == "done_success":
            return {"tunnel_id": tunnel_id, "slot": slot, "state": "done_success", "attempts": attempts}
    return {"tunnel_id": tunnel_id, "slot": slot, "state": "fail_after_retries", "attempts": attempts}


# ------------------------------------------------------------------------
# Dry-run path: no browser, no server. Loads the allowlist + bank files
# directly off disk and shows what would be dispatched.
# ------------------------------------------------------------------------

def load_bank_offline(tunnel_id: int, allowlist: set) -> list:
    """Mimic /api/human_bank/<tid>: load every completed human trace for the
    tunnel, filtered to the allowlist. Source-of-truth for the dry-run."""
    human_dir = REPO_ROOT / "data" / "human"
    out = []
    if not human_dir.is_dir():
        return out
    for fname in sorted(os.listdir(human_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(human_dir / fname) as f:
                d = json.load(f)
        except Exception:
            continue
        if not d.get("completed"):
            continue
        if d.get("tunnel_id") != tunnel_id:
            continue
        sid = d.get("session_id")
        if sid not in allowlist:
            continue
        out.append({
            "session_id": sid,
            "tunnel_id": d.get("tunnel_id"),
            "events": d.get("events", []),
        })
    return out


def dry_run(base_seed: int, max_attempts: int) -> None:
    print(f"=== replay_v2 dry-run ===")
    print(f"Allowlist: {ALLOWLIST_PATH}")
    allowlist = set(json.loads(ALLOWLIST_PATH.read_text()))
    print(f"  size: {len(allowlist)}")
    print(f"Base seed: {base_seed}  max_attempts: {max_attempts}")
    print()
    for tid in range(10):
        bank = load_bank_offline(tid, allowlist)
        if not bank:
            print(f"  tunnel {tid}: EMPTY BANK")
            continue
        # Slot 0, attempt 0: the canonical pick for this tunnel on round 1.
        seed = derive_seed(base_seed, tid, slot=0, attempt=0)
        choice = pick_source(bank, seed)
        events = choice["events"]
        slice_ = trim_to_active(events)
        first_three = [{"x": round(e["x"], 1), "y": round(e["y"], 1),
                        "t": round(e["timestamp"], 1), "et": e["event_type"]}
                       for e in slice_[:3]]
        last_three = [{"x": round(e["x"], 1), "y": round(e["y"], 1),
                       "t": round(e["timestamp"], 1), "et": e["event_type"]}
                      for e in slice_[-3:]]
        print(f"  tunnel {tid}: bank={len(bank)} seed={seed} pick={choice['session_id']}")
        print(f"    raw_events={len(events)} active_slice={len(slice_)}")
        print(f"    first 3: {first_three}")
        print(f"    last  3: {last_three}")
    print()
    print("Dry-run complete. No browser was launched, no server was hit.")


# ------------------------------------------------------------------------
# Live path
# ------------------------------------------------------------------------

async def run_live(args) -> None:
    from playwright.async_api import async_playwright

    out_dir = REPO_ROOT / "data" / "replay_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = out_dir / f"manifest_{ts}.json"

    manifest = {
        "experiment": "replay_v2",
        "timestamp": ts,
        "rounds": args.rounds,
        "max_attempts": args.max_attempts,
        "base_seed": args.seed,
        "headless": args.headless,
        "server_url": args.server,
        "allowlist_path": str(ALLOWLIST_PATH),
        "allowlist_sha": sha12(ALLOWLIST_PATH),
        "script_path": str(Path(__file__)),
        "script_sha": sha12(Path(__file__)),
        "git_sha": git_sha(),
        "rounds_data": [],
    }

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=args.headless)
        context = await browser.new_context()
        page = await context.new_page()

        # Verify server experiment mode.
        await page.goto(args.server)
        mode = await page.evaluate(
            "async () => (await (await fetch('/api/mode')).json())"
        )
        if mode.get("experiment") != "replay_v2":
            print(f"ERROR: server is in mode '{mode.get('experiment')}' but this run requires 'replay_v2'.")
            print("Restart the server with: python server.py --experiment replay_v2 "
                  "--allowed-sessions experiments_replay/allowed_sessions_v1b.json")
            await browser.close()
            sys.exit(2)

        # Wait for game.__tunnelGame to mount.
        await page.wait_for_function("() => !!window.__tunnelGame", timeout=10_000)

        for r in range(1, args.rounds + 1):
            print(f"=== Round {r} of {args.rounds} ===")
            round_results = []
            for tid in range(10):
                res = await run_tunnel(page, tid, slot=r - 1, base_seed=args.seed,
                                       max_attempts=args.max_attempts)
                print(f"  tunnel {tid}: {res['state']} "
                      f"(attempts={len(res['attempts'])})")
                round_results.append(res)
            manifest["rounds_data"].append({"round": r, "results": round_results})

        await browser.close()

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="replay_v2: per-tunnel bank replay via Playwright CDP mouse API."
    )
    ap.add_argument("--rounds", type=int, default=3, help="Number of replay rounds (default 3).")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="Max attempts per tunnel per round (default 3).")
    ap.add_argument("--seed", type=int, default=1,
                    help="Base seed for source-trace sampling (default 1).")
    ap.add_argument("--server", default=DEFAULT_SERVER,
                    help=f"Server URL (default {DEFAULT_SERVER}).")
    ap.add_argument("--headless", action="store_true",
                    help="Run Chromium headless. Default: headed (easier debugging).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip browser/server. Load the bank offline and print "
                         "which (tunnel, source) would be replayed.")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(base_seed=args.seed, max_attempts=args.max_attempts)
        return

    asyncio.run(run_live(args))


if __name__ == "__main__":
    main()
