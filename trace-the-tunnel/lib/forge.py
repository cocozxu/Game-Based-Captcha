"""
forge.py — read-side forge for the trace-the-tunnel captcha, extracted
from experiments_replay_init/replay_init_solver.py so other experiments
(e.g. experiments_generalize) can reuse the same channel.

What the forge does (verbatim from the original):
  - replaces MouseEvent.prototype.clientX / clientY getters to return
    queued source coordinates
  - replaces performance.now() to return the queued source timestamp
  - replaces Date.now() derived from performance.timeOrigin + now()
  - hides the overrides from Function.prototype.toString '[native code]'
    checks (defensive — current captcha does not check)

What this module exposes:
  - INIT_SCRIPT: the JS string to pass to playwright add_init_script
  - install_forge(context): convenience to install on a BrowserContext
  - dispatch_trace(cdp, page, events, geom): the busy-wait pacing + CDP
    dispatcher (with forge queue pre-population), unchanged behavior
  - get_canvas_geometry(page), load_tunnel(page, tid), wait_for_state(...)
    page helpers used by every solver that drives the captcha
  - cdp_type(event_type), slice_down_to_up(events) small utilities

Per-event read pattern in game.js (verified against static/game.js):
  getCanvasPos(e) reads e.clientX, e.clientY      (2 reads)
  recordEvent(...) reads performance.now()        (1 read)

Strategy:
  - clientX / clientY GETTERS only PEEK at the queue head
  - performance.now() POPS the head
This means all three reads for a given event see the same forged record,
then the queue advances. Robust to call order *within* one event handler.
"""

import asyncio
import time


INIT_SCRIPT = r"""
(() => {
  if (window.__forge) return;  // idempotent in case the page reloads

  const queue = [];
  const origClientX = Object.getOwnPropertyDescriptor(MouseEvent.prototype, "clientX").get;
  const origClientY = Object.getOwnPropertyDescriptor(MouseEvent.prototype, "clientY").get;
  const origNow     = performance.now.bind(performance);
  const origToString = Function.prototype.toString;
  const origDateNow = Date.now.bind(Date);

  const canvasRect = () => {
    const c = document.querySelector("canvas");
    return c ? c.getBoundingClientRect() : { left: 0, top: 0, width: 1, height: 1 };
  };
  const canvasLogicalSize = () => {
    const c = document.querySelector("canvas");
    return c ? { w: c.width, h: c.height } : { w: 1, h: 1 };
  };

  window.__forge = {
    push(ev) { queue.push(ev); },
    pushAll(arr) { for (const e of arr) queue.push(e); },
    clear() { queue.length = 0; },
    size() { return queue.length; },
    consumed: 0,
  };

  Object.defineProperty(MouseEvent.prototype, "clientX", {
    configurable: true,
    get() {
      if (queue.length === 0) return origClientX.call(this);
      const ev = queue[0];
      const r = canvasRect();
      const cs = canvasLogicalSize();
      // game.js does (clientX - rect.left) * (canvas_logical_w / rect.width)
      // we want that to equal ev.x, so:
      // clientX = ev.x * (rect.width / canvas_logical_w) + rect.left
      return ev.x * (r.width / cs.w) + r.left;
    },
  });
  Object.defineProperty(MouseEvent.prototype, "clientY", {
    configurable: true,
    get() {
      if (queue.length === 0) return origClientY.call(this);
      const ev = queue[0];
      const r = canvasRect();
      const cs = canvasLogicalSize();
      return ev.y * (r.height / cs.h) + r.top;
    },
  });

  performance.now = function () {
    if (queue.length === 0) return origNow();
    const ev = queue.shift();
    window.__forge.consumed++;
    return ev.t;
  };
  Date.now = function () {
    return performance.timeOrigin + performance.now();
  };

  // Light stealth: hide the override from `someFn.toString().includes('[native code]')`.
  // The current captcha does not check this; we do it so the experiment generalizes
  // to a captcha that *might*.
  Function.prototype.toString = function () {
    if (this === performance.now || this === Date.now) {
      return "function " + this.name + "() { [native code] }";
    }
    return origToString.call(this);
  };
})();
"""


async def install_forge(context):
    """Install the forge on a playwright BrowserContext. MUST be called
    before any page in that context navigates, otherwise game.js will
    capture the unmodified prototypes."""
    await context.add_init_script(INIT_SCRIPT)


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------

def cdp_type(event_type):
    return {"mousedown": "mousePressed", "mouseup": "mouseReleased", "mousemove": "mouseMoved"}[event_type]


def slice_down_to_up(events):
    down = next((i for i, e in enumerate(events) if e["event_type"] == "mousedown"), 0)
    up = next((i for i in range(len(events) - 1, -1, -1) if events[i]["event_type"] == "mouseup"), len(events) - 1)
    return events[down:up + 1]


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
    return await page.evaluate(
        "() => { const c = document.querySelector('canvas'); const r = c.getBoundingClientRect(); "
        "return {x: r.left, y: r.top, sx: r.width / c.width, sy: r.height / c.height}; }"
    )


# ---------------------------------------------------------------------------
# Dispatch — busy-wait pacing + CDP, with forge queue pre-population
# ---------------------------------------------------------------------------

async def dispatch_trace(cdp, page, events, geom):
    """Spin-wait pacing + fire-and-forget CDP. The forge handles
    correctness of the recorded values; we still need delivery pacing to
    prevent the renderer's coalescing pass from merging adjacent
    mousemoves (which would consume queue entries without firing a
    handler).

    `events`: list of dicts with keys (x, y, timestamp, event_type).
    Must include exactly one `mousedown` and one `mouseup`; only events
    between them are dispatched.

    Returns the number of events dispatched.
    """
    s = slice_down_to_up(events)
    if not s:
        return 0

    payload = [{"x": e["x"], "y": e["y"], "t": e["timestamp"]} for e in s]
    await page.evaluate("(payload) => { window.__forge.clear(); window.__forge.pushAll(payload); }", payload)

    src_origin = s[0]["timestamp"]
    cdp_origin_s = time.monotonic()
    start_ns = time.monotonic_ns()
    pending = []
    dispatched = 0

    for ev in s:
        et = ev["event_type"]
        if et not in ("mousedown", "mouseup", "mousemove"):
            continue

        target_ns = start_ns + int((ev["timestamp"] - src_origin) * 1e6)
        while time.monotonic_ns() < target_ns:
            pass

        vx = geom["x"] + ev["x"] * geom["sx"]
        vy = geom["y"] + ev["y"] * geom["sy"]
        ts = cdp_origin_s + (ev["timestamp"] - src_origin) / 1000.0

        params = {
            "type": cdp_type(et),
            "x": vx, "y": vy,
            "timestamp": ts,
            "button": "left",
            "buttons": 0 if et == "mouseup" else 1,
            "clickCount": 1 if et in ("mousedown", "mouseup") else 0,
        }
        pending.append(asyncio.ensure_future(cdp.send("Input.dispatchMouseEvent", params)))
        await asyncio.sleep(0)
        dispatched += 1

    await asyncio.gather(*pending, return_exceptions=True)
    return dispatched
