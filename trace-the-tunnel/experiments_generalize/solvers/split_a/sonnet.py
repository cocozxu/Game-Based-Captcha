"""Human-like trace generator for the trace-the-tunnel CAPTCHA (Sonnet 4.6).

Centerline: 4 cubic Bezier segments, 16 control points total (p0,p1,p2,p3
per segment with shared endpoint knots between consecutive segments).
Actual arc length per tunnel is typically 800-1100 px due to the S-curves.

Motion model:
  * Tanh-ramp speed profile: smooth acceleration over first ~60 px of arc,
    cruise at v_max, then smooth deceleration over last ~60 px. Combined
    with curvature-based slowdown (Fitts-like 1/(1+a*κ)).
  * Ornstein-Uhlenbeck perpendicular wobble: physically-motivated drift
    attracted back toward the centerline, avoiding pure periodic patterns.
  * Small multiplicative motor-noise velocity jitter added each step.
  * Event timing: ~8.3ms base dt (≈120 Hz) with Gaussian jitter;
    occasional micro-pauses matching human hesitation patterns.
  * Canvas-logical pixel offsets replicate the browser's sub-pixel transform.
  * Strictly monotonic timestamps; exactly one mousedown (first) and
    one mouseup (last at same timestamp as final mousemove).
"""

import math
import numpy as np

# Sub-pixel offsets observed uniformly in all human demo traces
X_OFFSET = 0.199981689453125
Y_OFFSET = 0.400001525878906


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _build_centerline(control_points, n_per_seg=350):
    """Sample dense points along the 4-segment cubic Bezier centerline."""
    pts = []
    for cp in control_points:
        if isinstance(cp, dict):
            pts.append([cp["x"], cp["y"]])
        else:
            pts.append([float(cp[0]), float(cp[1])])
    pts = np.array(pts, dtype=float)

    n_segs = len(pts) // 4
    chunks = []
    for i in range(n_segs):
        j = i * 4
        p0, p1, p2, p3 = pts[j], pts[j + 1], pts[j + 2], pts[j + 3]
        include_end = (i == n_segs - 1)
        n_pts = n_per_seg + (1 if include_end else 0)
        ts = np.linspace(0.0, 1.0, n_pts, endpoint=include_end)
        u = 1.0 - ts
        x = u**3 * p0[0] + 3*u**2*ts*p1[0] + 3*u*ts**2*p2[0] + ts**3*p3[0]
        y = u**3 * p0[1] + 3*u**2*ts*p1[1] + 3*u*ts**2*p2[1] + ts**3*p3[1]
        chunks.append(np.stack([x, y], axis=1))

    return np.concatenate(chunks, axis=0)


def _arc_lengths(path):
    """Cumulative arc lengths along a polyline."""
    diffs = np.diff(path, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def _curvature(path, win=15):
    """Smoothed discrete curvature (1/radius) at each point."""
    n = len(path)
    if n < 3:
        return np.zeros(n)
    v1 = path[1:-1] - path[:-2]
    v2 = path[2:] - path[1:-1]
    l1 = np.linalg.norm(v1, axis=1)
    l2 = np.linalg.norm(v2, axis=1)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    sin_t = np.clip(cross / np.maximum(l1 * l2, 1e-9), -1.0, 1.0)
    k = np.abs(np.arcsin(sin_t)) / np.maximum(l1, 1e-9)
    kappa = np.zeros(n)
    kappa[1:-1] = k
    kpad = np.pad(kappa, win, mode="edge")
    kernel = np.ones(2 * win + 1) / (2 * win + 1)
    return np.convolve(kpad, kernel, mode="same")[win:-win]


def _interp_at(path, arc, s):
    """Linear interpolation of path at arc-length position s."""
    s = float(np.clip(s, 0.0, arc[-1]))
    idx = int(np.searchsorted(arc, s, side="right"))
    idx = int(np.clip(idx, 1, len(arc) - 1))
    denom = max(arc[idx] - arc[idx - 1], 1e-9)
    f = (s - arc[idx - 1]) / denom
    return path[idx - 1] + f * (path[idx] - path[idx - 1])


def _perp_at(path, arc, s, eps=1.5):
    """Unit perpendicular to the path at arc-length s (90° CCW from tangent)."""
    L = float(arc[-1])
    p1 = _interp_at(path, arc, max(0.0, s - eps))
    p2 = _interp_at(path, arc, min(L, s + eps))
    d = p2 - p1
    nrm = math.hypot(d[0], d[1])
    if nrm < 1e-9:
        return np.array([0.0, 1.0])
    tg = d / nrm
    return np.array([-tg[1], tg[0]])


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def generate(tunnel_spec: dict, seed: int) -> list:
    """Generate a human-like event stream for the given tunnel."""
    rng = np.random.default_rng(int(seed))

    cps = tunnel_spec["control_points"]
    # game.js: distToCenterline <= TUNNEL_WIDTH=38, so half-boundary = 38 px
    boundary = float(tunnel_spec["tunnel_width"])

    path = _build_centerline(cps, n_per_seg=350)
    arc = _arc_lengths(path)
    total_len = float(arc[-1])
    kappa = _curvature(path, win=15)

    # --- Per-trace randomized parameters ---
    # Tunnels have arc length ~800-1100 px; humans traverse in ~1000-2500 ms
    # → effective average speed ~0.45-0.85 px/ms.
    # Mean curvature ~0.027; with curve_factor=10, slowdown ≈ 20% → v_cruise ≈ 0.75-1.1
    v_cruise = rng.uniform(0.70, 1.10)           # px/ms cruise speed
    curve_factor = rng.uniform(5.0, 18.0)        # curvature slowdown coefficient (mild)
    ramp_len = rng.uniform(50.0, 80.0)           # arc-length over which to ramp up/down
    dt_base = rng.uniform(8.22, 8.45)            # ms between events (~120 Hz)
    dt_sigma = rng.uniform(0.18, 0.38)           # jitter on dt
    # Ornstein-Uhlenbeck (OU) wobble: mean-reverting lateral drift
    ou_theta = rng.uniform(0.05, 0.15)           # reversion rate (1/px)
    ou_sigma = rng.uniform(0.35, 0.90)           # noise scale (px/sqrt(px))
    max_wobble = min(boundary * 0.55, 21.0)      # hard lateral clamp
    speed_jitter = rng.uniform(0.03, 0.09)       # per-step multiplicative noise

    def speed_at(s):
        """Tanh-ramp profile + curvature-based slowdown."""
        k = float(np.interp(s, arc, kappa))
        # Tanh ramps at start and end, cruise in between
        ramp_in = math.tanh(s / max(ramp_len, 1.0))
        ramp_out = math.tanh((total_len - s) / max(ramp_len, 1.0))
        ramp = math.sqrt(ramp_in * ramp_out)          # combined smooth envelope
        base = v_cruise * ramp
        curved = base / (1.0 + curve_factor * k)
        return max(0.04, curved)

    t0 = 3000.0 + rng.uniform(0.0, 580000.0)
    events = []

    p_start = path[0].copy()
    x0 = round(float(p_start[0])) + X_OFFSET
    y0 = round(float(p_start[1])) + Y_OFFSET

    # --- mousedown ---
    events.append({
        "x": x0, "y": y0,
        "timestamp": t0,
        "event_type": "mousedown",
        "inside_tunnel": True,
    })

    # First mousemove: same position, variable delay
    r = rng.random()
    if r < 0.12:
        first_gap = rng.uniform(0.5, 4.5)
    elif r < 0.90:
        first_gap = rng.uniform(7.5, 20.0)
    else:
        first_gap = rng.uniform(45.0, 160.0)

    t = t0 + first_gap
    events.append({
        "x": x0, "y": y0,
        "timestamp": t,
        "event_type": "mousemove",
        "inside_tunnel": True,
    })

    # --- Main traversal ---
    s = 0.0
    ou_state = 0.0   # current OU lateral offset (px)
    safety = 0

    while s < total_len - 0.4 and safety < 8000:
        safety += 1

        v = speed_at(s)
        # Multiplicative motor noise
        v *= (1.0 + rng.normal(0.0, speed_jitter))
        v = max(0.04, v)

        dt = rng.normal(dt_base, dt_sigma)
        if dt < 4.5:
            dt = 4.5
        # Rare micro-pause (~1.5% of events)
        if rng.random() < 0.015:
            dt += rng.uniform(3.0, 18.0)

        ds = v * dt
        s = min(total_len, s + ds)
        t += dt

        p = _interp_at(path, arc, s)
        perp = _perp_at(path, arc, s)

        # Ornstein-Uhlenbeck step: dx = -θ·x·ds + σ·√ds·N(0,1)
        ds_step = max(ds, 0.5)
        ou_state += (
            -ou_theta * ou_state * ds_step
            + ou_sigma * math.sqrt(ds_step) * rng.standard_normal()
        )
        ou_state = float(np.clip(ou_state, -max_wobble, max_wobble))

        p = p + perp * ou_state

        events.append({
            "x": round(float(p[0])) + X_OFFSET,
            "y": round(float(p[1])) + Y_OFFSET,
            "timestamp": t,
            "event_type": "mousemove",
            "inside_tunnel": True,
        })

    # --- Snap to endpoint if not already there ---
    p_end = path[-1]
    x_end = round(float(p_end[0])) + X_OFFSET
    y_end = round(float(p_end[1])) + Y_OFFSET
    last = events[-1]
    if last["x"] != x_end or last["y"] != y_end:
        dt = max(4.5, rng.normal(dt_base, dt_sigma))
        t += dt
        events.append({
            "x": x_end, "y": y_end,
            "timestamp": t,
            "event_type": "mousemove",
            "inside_tunnel": True,
        })

    # Optional end dwell (humans sometimes pause before releasing)
    if rng.random() < 0.32:
        n_dwell = int(rng.integers(1, 5))
        for _ in range(n_dwell):
            dt = max(4.5, rng.normal(dt_base, dt_sigma))
            t += dt
            events.append({
                "x": events[-1]["x"], "y": events[-1]["y"],
                "timestamp": t,
                "event_type": "mousemove",
                "inside_tunnel": True,
            })

    # --- mouseup at same timestamp as last move ---
    events.append({
        "x": events[-1]["x"], "y": events[-1]["y"],
        "timestamp": events[-1]["timestamp"],
        "event_type": "mouseup",
        "inside_tunnel": True,
    })

    return events
