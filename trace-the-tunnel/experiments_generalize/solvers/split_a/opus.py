"""Human-like trace generator for the trace-the-tunnel CAPTCHA.

The centerline is reconstructed as a chain of 4 cubic Bezier segments
(16 control points, 4 per segment with shared endpoints between
consecutive segments — verified from demo geometry & traces).

Motion model:
  * Cruise speed ~0.45-0.65 px/ms with eased start/end ramps.
  * Curvature-based slowdown at tight bends (Fitts-like 1/(1+a*k)).
  * Low-frequency perpendicular wobble (sine over arc length) keeps
    the trace off the exact mathematical centerline without leaving
    the tunnel boundary.
  * Sample dt drawn from N(8.3, 0.3) ms (~120Hz polling) with rare
    larger gaps; occasional initial/end dwells matching demo patterns.
  * x/y rounded to integer pixels + the constant canvas hotspot
    offsets observed across every demo trace.
"""
import math
import numpy as np

X_OFFSET = 0.199981689453125
Y_OFFSET = 0.400001525878906


def _bezier_seg(p0, p1, p2, p3, n, include_end):
    m = n + (1 if include_end else 0)
    ts = np.arange(m) / n
    u = 1.0 - ts
    x = u**3 * p0[0] + 3 * u**2 * ts * p1[0] + 3 * u * ts**2 * p2[0] + ts**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * ts * p1[1] + 3 * u * ts**2 * p2[1] + ts**3 * p3[1]
    return np.stack([x, y], axis=1)


def _build_centerline(control_points, n_per_seg=400):
    pts = []
    for cp in control_points:
        if isinstance(cp, dict):
            pts.append((cp['x'], cp['y']))
        else:
            pts.append((cp[0], cp[1]))
    n = len(pts)
    if n < 4:
        return np.array(pts, dtype=float)
    n_segs = n // 4
    if n_segs == 0:
        return np.array(pts, dtype=float)
    chunks = []
    for s_idx in range(n_segs):
        i = s_idx * 4
        seg = pts[i:i + 4]
        if len(seg) < 4:
            break
        last = (s_idx == n_segs - 1)
        chunks.append(_bezier_seg(seg[0], seg[1], seg[2], seg[3], n_per_seg, last))
    return np.concatenate(chunks, axis=0)


def _arc_length(path):
    diffs = np.diff(path, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _curvature_smooth(path, win=20):
    n = len(path)
    if n < 3:
        return np.zeros(n)
    v1 = path[1:-1] - path[:-2]
    v2 = path[2:] - path[1:-1]
    l1 = np.linalg.norm(v1, axis=1)
    l2 = np.linalg.norm(v2, axis=1)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    denom = np.maximum(l1 * l2, 1e-9)
    sin_t = np.clip(cross / denom, -1.0, 1.0)
    k = np.abs(np.arcsin(sin_t)) / np.maximum(l1, 1e-9)
    kappa = np.zeros(n)
    kappa[1:-1] = k
    kpad = np.pad(kappa, win, mode='edge')
    kernel = np.ones(2 * win + 1) / (2 * win + 1)
    return np.convolve(kpad, kernel, mode='same')[win:-win]


def _interp(path, arc, s):
    if s <= 0:
        return path[0].copy()
    if s >= arc[-1]:
        return path[-1].copy()
    idx = int(np.searchsorted(arc, s))
    if idx <= 0:
        return path[0].copy()
    if idx >= len(path):
        return path[-1].copy()
    s0, s1 = arc[idx - 1], arc[idx]
    f = (s - s0) / max(s1 - s0, 1e-9)
    return path[idx - 1] + f * (path[idx] - path[idx - 1])


def _tangent_at(path, arc, s, eps=1.0):
    L = arc[-1]
    p1 = _interp(path, arc, max(0.0, s - eps))
    p2 = _interp(path, arc, min(L, s + eps))
    d = p2 - p1
    n = math.hypot(d[0], d[1])
    if n < 1e-9:
        return np.array([1.0, 0.0])
    return d / n


def generate(tunnel_spec, seed):
    rng = np.random.default_rng(int(seed))
    cps = tunnel_spec['control_points']
    half_w = float(tunnel_spec['tunnel_width']) / 2.0

    path = _build_centerline(cps, n_per_seg=400)
    arc = _arc_length(path)
    total_len = float(arc[-1])
    kappa = _curvature_smooth(path, win=20)

    # Per-trace randomized parameters
    v_max = rng.uniform(0.42, 0.62)
    v_min = 0.15
    ramp_len = min(rng.uniform(35.0, 55.0), max(20.0, total_len / 4.0))
    curve_factor = rng.uniform(70.0, 115.0)
    dt_mean = rng.uniform(8.25, 8.45)
    dt_std = rng.uniform(0.20, 0.40)
    wobble_amp = rng.uniform(1.0, 3.5)
    wobble_freq = rng.uniform(0.03, 0.12)
    wobble_phase = rng.uniform(0.0, 2 * math.pi)
    max_wobble = max(1.5, half_w - 6.0)

    def speed_at(s):
        k = float(np.interp(s, arc, kappa))
        sp = v_max / (1.0 + curve_factor * k)
        return max(v_min, sp)

    t0 = 100000.0 + rng.uniform(0.0, 500000.0)
    events = []

    p_start = path[0]
    x_start = round(float(p_start[0])) + X_OFFSET
    y_start = round(float(p_start[1])) + Y_OFFSET

    events.append({
        "x": x_start, "y": y_start,
        "timestamp": t0,
        "event_type": "mousedown",
        "inside_tunnel": True,
    })

    # Variable first-move gap: matches the spread seen in demos
    r = rng.random()
    if r < 0.2:
        first_gap = rng.uniform(0.3, 5.0)
    elif r < 0.92:
        first_gap = rng.uniform(7.0, 25.0)
    else:
        first_gap = rng.uniform(50.0, 200.0)

    t = t0 + first_gap
    events.append({
        "x": x_start, "y": y_start,
        "timestamp": t,
        "event_type": "mousemove",
        "inside_tunnel": True,
    })

    # Occasionally a long dwell on the start point (rare initial pause)
    if first_gap < 30.0 and rng.random() < 0.08:
        t += rng.uniform(80.0, 250.0)
        events.append({
            "x": x_start, "y": y_start,
            "timestamp": t,
            "event_type": "mousemove",
            "inside_tunnel": True,
        })

    s = 0.0
    safety = 0
    while s < total_len - 0.3 and safety < 6000:
        safety += 1
        v = speed_at(s)
        ramp_in = min(1.0, (s + 3.0) / ramp_len)
        ramp_out = min(1.0, (total_len - s + 3.0) / ramp_len)
        ramp = math.sqrt(min(ramp_in, ramp_out))
        v_eff = max(0.06, v * ramp)

        dt = rng.normal(dt_mean, dt_std)
        if dt < 5.5:
            dt = 5.5
        if rng.random() < 0.02:
            dt += rng.uniform(2.0, 12.0)

        s = min(total_len, s + v_eff * dt)
        t += dt

        p = _interp(path, arc, s)
        tg = _tangent_at(path, arc, s)
        perp = np.array([-tg[1], tg[0]])
        w = wobble_amp * math.sin(wobble_freq * s + wobble_phase)
        if w > max_wobble:
            w = max_wobble
        elif w < -max_wobble:
            w = -max_wobble
        p = p + perp * w

        x = round(float(p[0])) + X_OFFSET
        y = round(float(p[1])) + Y_OFFSET

        events.append({
            "x": x, "y": y,
            "timestamp": t,
            "event_type": "mousemove",
            "inside_tunnel": True,
        })

    # Snap to exact endpoint pixel
    p_end = path[-1]
    x_end = round(float(p_end[0])) + X_OFFSET
    y_end = round(float(p_end[1])) + Y_OFFSET
    if events[-1]['x'] != x_end or events[-1]['y'] != y_end:
        dt = rng.normal(dt_mean, dt_std)
        t += max(5.5, dt)
        events.append({
            "x": x_end, "y": y_end,
            "timestamp": t,
            "event_type": "mousemove",
            "inside_tunnel": True,
        })

    # End dwell (some humans pause before releasing)
    if rng.random() < 0.35:
        n_dwell = int(rng.integers(1, 5))
        for _ in range(n_dwell):
            dt = rng.normal(dt_mean, dt_std)
            t += max(5.5, dt)
            events.append({
                "x": events[-1]['x'], "y": events[-1]['y'],
                "timestamp": t,
                "event_type": "mousemove",
                "inside_tunnel": True,
            })

    # mouseup at the same timestamp as the last move (matches demo)
    events.append({
        "x": events[-1]['x'], "y": events[-1]['y'],
        "timestamp": events[-1]['timestamp'],
        "event_type": "mouseup",
        "inside_tunnel": True,
    })

    return events
