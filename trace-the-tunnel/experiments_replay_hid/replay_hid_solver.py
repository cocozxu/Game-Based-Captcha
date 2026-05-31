"""
replay_hid_solver.py — replay attack via macOS HID-level mouse event injection.

Same source bank and per-trace loop as replay_cdp_solver.py. The only thing
that changes is the dispatch channel: instead of CDP `Input.dispatchMouseEvent`
(which queues through the renderer's input pipeline and inherits the
WebSocket/IPC jitter), this solver posts CGEvents directly into the macOS
HID event stream via Quartz. The renderer's IO thread sees these as real
mouse-device input, with hardware-stamped timestamps.

The goal: drive the residual `dt`-jitter PLAN.md identified (~0.1 ms,
amplified by jerk/tremor features) toward zero. If motor AUC drops
significantly relative to replay_cdp, we've confirmed the (good motor,
good mechanism) corner is reachable by an attacker with kernel-level
input-injection capability.

Requirements:
  - macOS (Quartz is Darwin-only). Aborts on other platforms.
  - Accessibility permission for the running Python / terminal. The script
    runs a probe at startup and prints an actionable error if the prompt
    needs to be granted (System Settings → Privacy & Security → Accessibility).
  - Headed browser only — CGEventPost requires a live windowserver session
    and a foreground window for events to land predictably. Headless
    Chromium has no real window for events to target.

The captcha server must run as:
  python server.py --experiment replay_hid --expose-debug \\
    --allowed-sessions experiments_replay/allowed_sessions_v1b.json
"""

import argparse
import asyncio
import json
import os
import platform
import random
import statistics
import sys
import time

import requests

if platform.system() != "Darwin":
    print("replay_hid_solver requires macOS (Quartz CGEventPost).", file=sys.stderr)
    sys.exit(2)

try:
    import Quartz
except ImportError:
    print("Quartz module missing. Install with: pip install pyobjc-framework-Quartz", file=sys.stderr)
    sys.exit(2)

from playwright.async_api import async_playwright


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(REPO, "data")
OUT_DIR = os.path.join(DATA_DIR, "replay_hid")

# Hard cap on dispatched events per trace. Source traces are typically
# 150–200 events; this is generous. A misconfigured loop with a runaway
# spinner would otherwise spew clicks across the screen.
MAX_EVENTS_PER_TRACE = 1000


# ---------------------------------------------------------------------------
# Source-trace handling (shared with replay_cdp)
# ---------------------------------------------------------------------------

def slice_down_to_up(events):
    down = next((i for i, e in enumerate(events) if e["event_type"] == "mousedown"), 0)
    up = next((i for i in range(len(events) - 1, -1, -1) if events[i]["event_type"] == "mouseup"), len(events) - 1)
    return events[down:up + 1]


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
# Quartz CGEvent dispatch
# ---------------------------------------------------------------------------

def cg_event_type(event_type, button_down):
    """Pick the right CGEvent type. Critical detail: while the button is
    pressed, the OS expects LeftMouseDragged for moves, not MouseMoved.
    Chrome will still surface them as 'mousemove' to JS, but the input
    pipeline treats them differently and skipping the dragged variant can
    cause Chrome to drop events or coalesce them oddly."""
    if event_type == "mousedown":
        return Quartz.kCGEventLeftMouseDown
    if event_type == "mouseup":
        return Quartz.kCGEventLeftMouseUp
    # mousemove
    return Quartz.kCGEventLeftMouseDragged if button_down else Quartz.kCGEventMouseMoved


def post_event(kind, x, y):
    """Post a single mouse event at global screen coords (x, y) in points.
    Returns immediately; CGEventPost is synchronous but very cheap (~µs)."""
    e = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, e)


async def calibrate_viewport_origin(page):
    """Find where the page's viewport (0, 0) lives in macOS screen coords.

    Computing it from `window.screenX/Y` + `outerHeight - innerHeight` is
    off by a few pixels on macOS Chrome (top status area, asymmetric chrome
    height). The reliable approach is to post a CGEvent at a known screen
    point, listen for it in JS, and subtract.

    Returns a (sx0, sy0) tuple such that
        screen(x, y) = (sx0 + viewport_x, sy0 + viewport_y).
    """
    sx = await page.evaluate("() => window.screenX")
    sy = await page.evaluate("() => window.screenY")
    # Install a one-shot probe at the document level so the calibration
    # event lands on something even if the canvas isn't visible.
    await page.evaluate("""() => {
      window.__hidProbe = new Promise((resolve) => {
        const h = (e) => { document.removeEventListener('mousemove', h, true); resolve({x: e.clientX, y: e.clientY}); };
        document.addEventListener('mousemove', h, true);
      });
    }""")
    # Pick a screen point that's *probably* inside the viewport. Use the
    # JS-reported screenX/Y as a starting estimate plus enough offset to
    # land well inside the page.
    test_sx, test_sy = sx + 200, sy + 220
    post_event(Quartz.kCGEventMouseMoved, test_sx, test_sy)
    try:
        client = await asyncio.wait_for(
            page.evaluate("() => window.__hidProbe"),
            timeout=2.0,
        )
    except asyncio.TimeoutError:
        raise RuntimeError(
            "Calibration probe never reached the page. Most likely the "
            "running process does not have Accessibility permission. "
            "Open System Settings → Privacy & Security → Accessibility and "
            "enable the app running this script (Terminal / iTerm / VSCode)."
        )
    sx0 = test_sx - client["x"]
    sy0 = test_sy - client["y"]
    return sx0, sy0


async def wait_for_state(page, target, timeout_ms=3000, poll_ms=50):
    deadline = time.monotonic() + timeout_ms / 1000.0
    last = None
    while time.monotonic() < deadline:
        last = await page.evaluate(
            "() => (window.__tunnelGame && window.__tunnelGame.getState) ? window.__tunnelGame.getState() : null"
        )
        if last in target:
            return last
        await asyncio.sleep(poll_ms / 1000.0)
    return last


async def load_tunnel(page, tid):
    ok = await page.evaluate(f"() => window.__tunnelGame.loadTunnel({tid})")
    if not ok:
        raise RuntimeError(f"loadTunnel({tid}) returned false")
    st = await wait_for_state(page, ("ready",), timeout_ms=2000)
    if st != "ready":
        raise RuntimeError(f"tunnel {tid}: state={st} after loadTunnel")


async def get_canvas_geometry(page):
    """Canvas position in viewport coords (CSS pixels)."""
    return await page.evaluate(
        "() => { const c = document.querySelector('canvas'); const r = c.getBoundingClientRect(); "
        "return {x: r.left, y: r.top, sx: r.width / c.width, sy: r.height / c.height}; }"
    )


def dispatch_trace_hid(events, sx0, sy0, geom):
    """Spin-wait + sync CGEventPost. No await between events — CGEventPost
    is a sync userspace call (<1 µs typical), so the spin-wait controls
    pacing directly and there's no async hop to schedule around. Runs to
    completion in ~1.3 s and blocks the asyncio loop during that time."""
    s = slice_down_to_up(events)
    if not s:
        return 0
    if len(s) > MAX_EVENTS_PER_TRACE:
        raise RuntimeError(f"trace has {len(s)} events, above MAX_EVENTS_PER_TRACE={MAX_EVENTS_PER_TRACE}")

    src_origin = s[0]["timestamp"]
    start_ns = time.monotonic_ns()
    button_down = False
    dispatched = 0

    for i, ev in enumerate(s):
        et = ev["event_type"]
        if et not in ("mousedown", "mouseup", "mousemove"):
            continue

        target_ns = start_ns + int((ev["timestamp"] - src_origin) * 1e6)
        while time.monotonic_ns() < target_ns:
            pass

        # canvas-logical (x, y) → viewport CSS pixels → global screen points
        vx = geom["x"] + ev["x"] * geom["sx"]
        vy = geom["y"] + ev["y"] * geom["sy"]
        screen_x = sx0 + vx
        screen_y = sy0 + vy

        kind = cg_event_type(et, button_down)
        post_event(kind, screen_x, screen_y)
        if et == "mousedown":
            button_down = True
        elif et == "mouseup":
            button_down = False
        dispatched += 1

    return dispatched


async def run_one(page, tid, source_trace, sx0, sy0):
    await load_tunnel(page, tid)
    geom = await get_canvas_geometry(page)
    n = dispatch_trace_hid(source_trace["events"], sx0, sy0, geom)
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
        browser = await p.chromium.launch(
            headless=False,
            args=[f"--window-position={args.window_x},{args.window_y}",
                  f"--window-size={args.window_w},{args.window_h}"],
        )
        ctx = await browser.new_context(viewport={"width": args.window_w, "height": args.window_h})
        page = await ctx.new_page()
        await page.goto(args.server + "/", wait_until="load")
        await page.wait_for_function("() => !!window.__tunnelGame", timeout=10000)
        await page.bring_to_front()
        await asyncio.sleep(0.8)  # let the window actually become foreground

        print("Calibrating viewport origin via probe event...")
        sx0, sy0 = await calibrate_viewport_origin(page)
        print(f"  viewport (0,0) lives at screen ({sx0}, {sy0})")

        banks = {}
        for tid in range(10):
            banks[tid] = fetch_bank(args.server, tid)
            print(f"[bank] tunnel {tid}: {len(banks[tid])} sources")

        results = []
        manifest_started = time.time()
        for tid in range(10):
            bank = banks[tid]
            if not bank:
                print(f"[tunnel {tid}] empty bank, skipping")
                continue
            slot = 0
            failures = 0
            while slot < args.per_tunnel and failures < 4 * args.per_tunnel:
                choice = rng.choice(bank)
                print(f"[tunnel {tid} slot {slot}] source={choice['session_id'][:8]} events={len(choice['events'])}", flush=True)
                before = set(os.listdir(OUT_DIR))
                # Make sure the browser is still foreground before dispatch
                # — if focus shifted (e.g., notification stole it), events
                # would land elsewhere. bring_to_front is cheap.
                await page.bring_to_front()
                try:
                    r = await run_one(page, tid, choice, sx0, sy0)
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
                        print(f"  -> SUCCESS saved={saved_name}", flush=True)
                        results.append({
                            "tunnel_id": tid,
                            "slot": slot,
                            "source_session": choice["session_id"],
                            "saved": saved_name,
                            "n_dispatched": r["n_dispatched"],
                        })
                        slot += 1
                    else:
                        print(f"  -> done_success but no file appeared", flush=True)
                        failures += 1
                else:
                    print(f"  -> state={r['state']} n_dispatched={r['n_dispatched']}", flush=True)
                    failures += 1

        n_saved = sum(1 for r in results if r.get("saved"))
        manifest = {
            "experiment": "replay_hid",
            "started": manifest_started,
            "finished": time.time(),
            "per_tunnel": args.per_tunnel,
            "seed": args.seed,
            "n_saved": n_saved,
            "results": results,
            "viewport_origin_screen": [sx0, sy0],
        }
        manifest_path = os.path.join(OUT_DIR, f"manifest_{int(manifest_started)}.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nSaved {n_saved} successful traces. Manifest: {manifest_path}")

        await browser.close()


# ---------------------------------------------------------------------------
# Validate mode (mirrors replay_cdp_solver §4.3 checks)
# ---------------------------------------------------------------------------

async def cmd_validate(args):
    print("=" * 60)
    print("Validation checks (HID dispatcher)")
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

    # ---- Check 1: aggregate dt drift ----
    print("\n--- Check 1: aggregate dt vs source ---")
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
    print(f"  n compared = {len(dms)}")
    if dms:
        print(f"  |Δdt_mean| median={statistics.median(dms):.4f} ms max={max(dms):.4f} ms")
        print(f"  |Δdt_std|  median={statistics.median(dss):.4f} ms max={max(dss):.4f} ms")
        if statistics.median(dms) <= 0.1 and statistics.median(dss) <= 0.3:
            print("  PASS (median within target)")
        else:
            print("  FAIL")
            overall_ok = False

    # ---- Check 2: isTrusted ----
    print("\n--- Check 2: event.isTrusted on CGEvent-dispatched mouse event ---")
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=[f"--window-position={args.window_x},{args.window_y}",
                  f"--window-size={args.window_w},{args.window_h}"],
        )
        ctx = await browser.new_context(viewport={"width": args.window_w, "height": args.window_h})
        page = await ctx.new_page()
        await page.goto(args.server + "/", wait_until="load")
        await page.wait_for_function("() => !!window.__tunnelGame", timeout=10000)
        await page.bring_to_front()
        await asyncio.sleep(0.8)
        sx0, sy0 = await calibrate_viewport_origin(page)
        await page.evaluate("""() => {
          window.__trustProbe = new Promise((resolve) => {
            const c = document.querySelector('canvas');
            const h = (e) => { c.removeEventListener('mousemove', h, true); resolve(e.isTrusted); };
            c.addEventListener('mousemove', h, true);
          });
        }""")
        geom = await get_canvas_geometry(page)
        post_event(Quartz.kCGEventMouseMoved, sx0 + geom["x"] + 300 * geom["sx"], sy0 + geom["y"] + 175 * geom["sy"])
        is_trusted = await page.evaluate("() => window.__trustProbe")
        print(f"  event.isTrusted = {is_trusted}")
        if is_trusted is True:
            print("  PASS")
        else:
            print("  FAIL")
            overall_ok = False
        await browser.close()

    # ---- Check 3: (x, y) match source ----
    print("\n--- Check 3: aggregate (x, y) drift ---")
    all_dx = []
    all_dy = []
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
        print(f"  |Δx| mean={statistics.fmean(all_dx):.3f} px max={max(all_dx):.3f} px")
        print(f"  |Δy| mean={statistics.fmean(all_dy):.3f} px max={max(all_dy):.3f} px")
        if statistics.fmean(all_dx) < 1.5 and statistics.fmean(all_dy) < 1.5:
            print("  PASS")
        else:
            print("  FAIL")
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
    ap.add_argument("--window-x", type=int, default=100)
    ap.add_argument("--window-y", type=int, default=100)
    ap.add_argument("--window-w", type=int, default=1024)
    ap.add_argument("--window-h", type=int, default=768)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    if args.validate_only:
        sys.exit(asyncio.run(cmd_validate(args)))
    else:
        asyncio.run(cmd_collect(args))


if __name__ == "__main__":
    main()
