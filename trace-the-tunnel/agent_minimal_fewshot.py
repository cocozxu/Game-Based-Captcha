"""
agent_minimal_fewshot.py

Few-shot human-mimicking trace generator for trace-the-tunnel.

Design notes (derived from experiments/examples/example-{1,2,3}.png):
  Real human players DO NOT trace the wavy centerline. They produce a much
  smoother path that "cuts corners" — riding inside every bend, only
  deviating from straight when a hard turn forces them to. The dashed
  centerline visible in the tunnel art is largely ignored.

Algorithm:
  1. Replicate the game's tunnel generation from seed (mulberry32 + chained
     cubic Beziers), so we have the exact centerline + control points the
     server will see.
  2. Heavy Gaussian smoothing on the centerline (sigma scales with tunnel
     half-width). This is the corner-cut: a low-pass filter on the wiggle.
  3. Boundary safety: any smoothed point further than (half_width - margin)
     from the centerline is pulled back toward the nearest centerline point.
  4. Re-sample the safe path by arc length (~4 px steps), matching the
     median step size measured from successful human traces.
  5. Add micro-jitter (~0.5 px Gaussian) — humans aren't perfectly steady.
  6. Generate event timestamps with an ease-in/ease-out cosine schedule and
     ~1200-1400 ms total duration.
  7. POST a session JSON straight to the running server at /api/save_trajectory.
     The server is in --experiment agent_minimal_fewshot mode, so it forces
     the source field and routes the file to data/agent_minimal_fewshot/.

Run:
  source .venv/bin/activate && python agent_minimal_fewshot.py
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import uuid
from typing import List, Tuple

import urllib.request

# ---------------------------------------------------------------------------
# Game constants — must match static/game.js
# ---------------------------------------------------------------------------

CANVAS_W = 600
CANVAS_H = 350
TUNNEL_HALF_WIDTH = 38            # game.js: TUNNEL_WIDTH (it's a half-width)
NUM_SEGMENTS = 4
SAMPLES_PER_SEG = 80
DOT_RADIUS = 14

SERVER_URL = "http://localhost:5050"
TUNNEL_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "tunnel_config.json"
)

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Tunnel generation — Python port of the JS in static/game.js
# ---------------------------------------------------------------------------

def mulberry32(seed: int):
    """Bit-exact port of game.js mulberry32. Yields floats in [0, 1)."""
    state = seed & 0xFFFFFFFF

    def _next() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = imul(state ^ (state >> 15), 1 | state) & 0xFFFFFFFF
        t = (t + imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF
        t = (t ^ t) & 0xFFFFFFFF  # JS: t = ... ^ t
        # The JS line is: t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        # (t ^ t) is always 0 — that line in the JS is a typo-style identity;
        # the real return uses a different t. Replicate exactly:
        return (((t ^ (t >> 14)) & 0xFFFFFFFF) >> 0) / 4294967296.0

    return _next


def imul(a: int, b: int) -> int:
    """Math.imul: 32-bit signed multiplication, then back to JS-style int."""
    a &= 0xFFFFFFFF
    b &= 0xFFFFFFFF
    # Convert to signed 32 to mimic JS Math.imul rounding semantics
    if a >= 0x80000000:
        a -= 0x100000000
    if b >= 0x80000000:
        b -= 0x100000000
    return (a * b) & 0xFFFFFFFF


def mulberry32_exact(seed: int):
    """Direct line-by-line port of game.js mulberry32."""
    state = seed & 0xFFFFFFFF

    def to_int32(x: int) -> int:
        x &= 0xFFFFFFFF
        return x - 0x100000000 if x >= 0x80000000 else x

    def _next() -> float:
        nonlocal state
        # seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
        state = to_int32(state) | 0
        state = to_int32(state + 0x6D2B79F5) | 0
        # let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        s_u = state & 0xFFFFFFFF
        t = imul(s_u ^ (s_u >> 15), 1 | s_u)
        # t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        t_signed = to_int32(t)
        inner = imul(t ^ (t >> 7), 61 | t)
        t = (to_int32(t_signed + to_int32(inner))) ^ t_signed
        # return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        t_u = t & 0xFFFFFFFF
        return ((t_u ^ (t_u >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return _next


def cubic_bezier(seg, t: float) -> Point:
    p0, p1, p2, p3 = seg
    u = 1 - t
    x = (
        u * u * u * p0[0]
        + 3 * u * u * t * p1[0]
        + 3 * u * t * t * p2[0]
        + t * t * t * p3[0]
    )
    y = (
        u * u * u * p0[1]
        + 3 * u * u * t * p1[1]
        + 3 * u * t * t * p2[1]
        + t * t * t * p3[1]
    )
    return (x, y)


def generate_tunnel(seed: int):
    """Replicates game.js generateTunnel(). Returns (control_points, centerline)."""
    rng = mulberry32_exact(seed)
    margin = 50
    usable_h = CANVAS_H - 2 * margin

    knots: List[Point] = []
    for i in range(NUM_SEGMENTS + 1):
        frac = i / NUM_SEGMENTS
        x = margin + frac * (CANVAS_W - 2 * margin)
        y = margin + rng() * usable_h
        knots.append((x, y))

    segments = []
    prev_handle = None
    for i in range(NUM_SEGMENTS):
        a = knots[i]
        b = knots[i + 1]
        span_y = usable_h * 0.7

        if prev_handle is not None:
            cp1 = (2 * a[0] - prev_handle[0], 2 * a[1] - prev_handle[1])
        else:
            d = -1 if rng() < 0.5 else 1
            cp1 = (
                a[0] + (b[0] - a[0]) * 0.3,
                a[1] + d * (0.3 + rng() * 0.5) * span_y,
            )

        mid_y = (a[1] + b[1]) / 2
        d2 = 1 if cp1[1] < mid_y else -1
        cp2 = (
            a[0] + (b[0] - a[0]) * 0.7,
            b[1] + d2 * (0.3 + rng() * 0.5) * span_y,
        )

        # Clamp to canvas bounds
        cp1 = (cp1[0], max(margin * 0.3, min(CANVAS_H - margin * 0.3, cp1[1])))
        cp2 = (cp2[0], max(margin * 0.3, min(CANVAS_H - margin * 0.3, cp2[1])))

        segments.append((a, cp1, cp2, b))
        prev_handle = cp2

    # Flat control point list (matches the JSON shape — duplicates included)
    control_points = []
    for seg in segments:
        for p in seg:
            control_points.append({"x": p[0], "y": p[1]})

    # Sample centerline; first point of subsequent segs is skipped as in JS
    centerline: List[Point] = []
    for seg in segments:
        for i in range(SAMPLES_PER_SEG + 1):
            if centerline and i == 0:
                continue
            t = i / SAMPLES_PER_SEG
            centerline.append(cubic_bezier(seg, t))

    return control_points, centerline


# ---------------------------------------------------------------------------
# Path algorithm — the human-mimicking part
# ---------------------------------------------------------------------------

def gaussian_kernel(sigma: float) -> List[float]:
    radius = max(1, int(math.ceil(sigma * 3)))
    weights = [math.exp(-(i * i) / (2 * sigma * sigma)) for i in range(-radius, radius + 1)]
    s = sum(weights)
    return [w / s for w in weights]


def smooth_path(points: List[Point], sigma: float) -> List[Point]:
    kernel = gaussian_kernel(sigma)
    radius = (len(kernel) - 1) // 2
    out: List[Point] = []
    for i in range(len(points)):
        sx = sy = 0.0
        for k, w in enumerate(kernel):
            j = max(0, min(len(points) - 1, i + k - radius))
            sx += points[j][0] * w
            sy += points[j][1] * w
        out.append((sx, sy))
    # Pin endpoints — never let smoothing pull start/end off the dots
    out[0] = points[0]
    out[-1] = points[-1]
    return out


def nearest_centerline_index(p: Point, centerline: List[Point]) -> Tuple[int, float]:
    best, best_d = 0, float("inf")
    for i, c in enumerate(centerline):
        d = math.hypot(p[0] - c[0], p[1] - c[1])
        if d < best_d:
            best_d, best = d, i
    return best, best_d


def clamp_to_tunnel(
    path: List[Point], centerline: List[Point], half_width: float, margin: float
) -> List[Point]:
    safe = half_width - margin
    out: List[Point] = []
    for p in path:
        idx, d = nearest_centerline_index(p, centerline)
        if d <= safe:
            out.append(p)
            continue
        c = centerline[idx]
        t = safe / d
        out.append((c[0] + (p[0] - c[0]) * t, c[1] + (p[1] - c[1]) * t))
    return out


def resample_by_arc_length(path: List[Point], step_px: float) -> List[Point]:
    if len(path) < 2:
        return list(path)
    out: List[Point] = [path[0]]
    carry = 0.0
    for i in range(1, len(path)):
        a, b = path[i - 1], path[i]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg == 0:
            continue
        traveled = -carry
        while traveled + step_px <= seg:
            traveled += step_px
            t = traveled / seg
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        carry = seg - traveled
    last = path[-1]
    if math.hypot(out[-1][0] - last[0], out[-1][1] - last[1]) > 0.5:
        out.append(last)
    return out


def add_jitter(path: List[Point], std_px: float, rng: random.Random) -> List[Point]:
    out: List[Point] = []
    for i, p in enumerate(path):
        if i == 0 or i == len(path) - 1:
            out.append(p)
        else:
            out.append((p[0] + rng.gauss(0, std_px), p[1] + rng.gauss(0, std_px)))
    return out


def ease_schedule(n: int, total_ms: float) -> List[float]:
    """Cumulative ms timestamps for n waypoints, cosine ease-in/ease-out."""
    times: List[float] = []
    for i in range(n):
        t = i / max(1, n - 1)
        eased = 0.5 - 0.5 * math.cos(math.pi * t)
        times.append(eased * total_ms)
    return times


def build_human_path(
    centerline: List[Point],
    half_width: float,
    rng: random.Random,
    *,
    sigma: float | None = None,
    margin: float = 12.0,
    step_px: float = 4.0,
    jitter_px: float = 0.5,
) -> List[Point]:
    if sigma is None:
        sigma = max(8.0, half_width * 0.55)
    path = smooth_path(centerline, sigma)
    path = clamp_to_tunnel(path, centerline, half_width, margin)
    path = resample_by_arc_length(path, step_px)
    path = add_jitter(path, jitter_px, rng)
    return path


# ---------------------------------------------------------------------------
# Validation — the same predicate game.js uses to decide pass/fail
# ---------------------------------------------------------------------------

def dist_to_segment(p: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def dist_to_centerline_polyline(p: Point, centerline: List[Point]) -> float:
    best = float("inf")
    for i in range(len(centerline) - 1):
        d = dist_to_segment(p, centerline[i], centerline[i + 1])
        if d < best:
            best = d
    return best


# ---------------------------------------------------------------------------
# Session construction & POST
# ---------------------------------------------------------------------------

def build_session(
    tunnel_id: int,
    tunnel_seed: int,
    control_points,
    centerline: List[Point],
    waypoints: List[Point],
    total_ms: float,
) -> dict:
    """Construct the session JSON the server expects."""
    # Browser-style monotonic clock, jittered slightly
    t0 = time.perf_counter() * 1000.0
    timings = ease_schedule(len(waypoints), total_ms)

    events = []
    boundary_violations = 0
    for i, ((x, y), t) in enumerate(zip(waypoints, timings)):
        ts = t0 + t
        inside = dist_to_centerline_polyline((x, y), centerline) <= TUNNEL_HALF_WIDTH
        if i == 0:
            etype = "mousedown"
        elif i == len(waypoints) - 1:
            etype = "mouseup"
        else:
            etype = "mousemove"
            if not inside:
                boundary_violations += 1
        events.append({
            "x": round(x, 1),
            "y": round(y, 1),
            "timestamp": ts,
            "event_type": etype,
            "inside_tunnel": bool(inside),
        })

    completed = (boundary_violations == 0)
    fail_reason = None if completed else "out_of_bounds"

    return {
        "session_id": str(uuid.uuid4()),
        "tunnel_id": tunnel_id,
        "tunnel_seed": tunnel_seed,
        "control_points": control_points,
        "tunnel_width": TUNNEL_HALF_WIDTH,
        "canvas_size": {"width": CANVAS_W, "height": CANVAS_H},
        "source": "agent_minimal_fewshot",
        "completed": completed,
        "fail_reason": fail_reason,
        "boundary_violations": boundary_violations,
        "start_time": events[0]["timestamp"],
        "end_time": events[-1]["timestamp"],
        "duration_ms": events[-1]["timestamp"] - events[0]["timestamp"],
        "viewport": {"width": 1200, "height": 800},
        "events": events,
    }


def post_session(session: dict) -> dict:
    body = json.dumps(session).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/api/save_trajectory",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def play_one(tunnel_cfg: dict, rng: random.Random) -> dict:
    tunnel_id = tunnel_cfg["tunnel_id"]
    seed = tunnel_cfg["seed"]

    control_points, centerline = generate_tunnel(seed)

    # Anchor path to the actual start/end so the synthetic mousedown sits on
    # the green dot (within DOT_RADIUS+4 = 18 px). centerline[0] is exactly
    # the start knot; we keep it.
    path = build_human_path(
        centerline,
        TUNNEL_HALF_WIDTH,
        rng,
        margin=12.0,
        step_px=4.0,
        jitter_px=0.5,
    )

    total_ms = 1200 + rng.random() * 250  # 1200-1450 ms
    session = build_session(
        tunnel_id, seed, control_points, centerline, path, total_ms
    )

    result = post_session(session)
    return {
        "tunnel_id": tunnel_id,
        "session_id": session["session_id"],
        "events": len(session["events"]),
        "completed": session["completed"],
        "duration_ms": round(session["duration_ms"], 1),
        "boundary_violations": session["boundary_violations"],
        "server": result,
    }


def main():
    with open(TUNNEL_CONFIG_PATH) as f:
        cfg = json.load(f)

    # Per-tunnel seeded RNG so noise is reproducible per tunnel
    summary = []
    for tun in cfg["tunnels"]:
        rng = random.Random(0xBEEF ^ tun["tunnel_id"])
        result = play_one(tun, rng)
        print(
            f"tunnel {result['tunnel_id']}: "
            f"events={result['events']:>4}  "
            f"duration={result['duration_ms']:>6.1f}ms  "
            f"completed={result['completed']}  "
            f"bv={result['boundary_violations']}  "
            f"-> {result['server']}"
        )
        summary.append(result)

    n_ok = sum(1 for r in summary if r["completed"])
    print(f"\n{n_ok}/{len(summary)} tunnels completed cleanly")


if __name__ == "__main__":
    main()
