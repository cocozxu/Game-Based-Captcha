"""Generalization solver for the trace-the-tunnel CAPTCHA.

Approach
--------
1. Reconstruct the centerline. The 16 ``control_points`` are 4 cubic
   Bezier segments laid out flat (each segment is 4 consecutive points
   and consecutive segments share an endpoint, which is why indices
   3=4, 7=8, 11=12 are duplicated). Sample each segment densely.
2. Build a speed profile that slows in regions of high local curvature.
   Humans obey something close to the 2/3 power law on free hand drag
   trajectories, so tangential speed drops where curvature is high.
3. Add a per-trace base-speed and ease-in/ease-out at the endpoints
   (people accelerate from the start, brake into the goal).
4. Walk along the centerline at variable speed sampled at ~8.3 ms
   intervals with Gaussian dt jitter and occasional micro-pauses;
   linear-interpolate position by arc length.
5. Add a smoothly varying small perpendicular offset (a few px, low
   frequency) so the path does not sit exactly on the analytical
   centerline.
6. Round (x, y) to integer pixels plus a per-trace constant subpixel
   offset, mimicking the page->canvas transform artefact humans show.
7. Emit one ``mousedown`` at the start, an initial pause (~100-200 ms),
   then ``mousemove``s with strictly monotonic timestamps, and a
   ``mouseup`` at the final position.
"""

import numpy as np


def _build_centerline(control_points, samples_per_segment=400):
    cps = np.array([(cp["x"], cp["y"]) for cp in control_points], dtype=float)
    n_seg = len(cps) // 4
    pts = []
    for s in range(n_seg):
        p0, p1, p2, p3 = cps[s * 4 : (s + 1) * 4]
        for j in range(samples_per_segment):
            t = j / samples_per_segment
            mt = 1.0 - t
            pt = (
                (mt * mt * mt) * p0
                + (3 * mt * mt * t) * p1
                + (3 * mt * t * t) * p2
                + (t * t * t) * p3
            )
            pts.append(pt)
    pts.append(cps[-1])
    return np.array(pts)


def _smooth_curvature(path, window=15):
    n = len(path)
    curv = np.zeros(n)
    w = min(window, max(1, n // 6))
    for i in range(w, n - w):
        v1 = path[i] - path[i - w]
        v2 = path[i + w] - path[i]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 > 1e-6 and n2 > 1e-6:
            cos_t = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            curv[i] = float(np.arccos(cos_t)) / max(n1 + n2, 1e-6)
    if n > 21:
        k = np.ones(21) / 21.0
        curv = np.convolve(curv, k, mode="same")
    return curv


def generate(tunnel_spec, seed):
    rng = np.random.default_rng(int(seed))

    path = _build_centerline(tunnel_spec["control_points"])
    n = len(path)

    diffs = np.diff(path, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(cumlen[-1])
    if total_len <= 0:
        return []

    curv = _smooth_curvature(path)
    base_speed = float(rng.uniform(0.55, 0.85))  # px/ms in straights
    speed = base_speed / (1.0 + 60.0 * curv)
    speed = np.maximum(speed, 0.12)

    ease = min(80, n // 8)
    if ease > 0:
        for i in range(ease):
            f = 0.30 + 0.70 * (i / ease)
            speed[i] *= f
            speed[-(i + 1)] *= f

    tangents = np.zeros_like(path)
    tangents[1:-1] = path[2:] - path[:-2]
    tangents[0] = path[1] - path[0]
    tangents[-1] = path[-1] - path[-2]
    tnorm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tnorm[tnorm < 1e-6] = 1.0
    tangents = tangents / tnorm
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    n_knots = max(6, n // 80)
    knot_x = np.linspace(0.0, n - 1, n_knots)
    knot_y = rng.uniform(-2.5, 2.5, n_knots)
    knot_y[0] *= 0.3
    knot_y[-1] *= 0.3
    jitter_amp = np.interp(np.arange(n), knot_x, knot_y)
    jittered = path + normals * jitter_amp[:, None]

    x_offset = float(rng.choice([0.0, 0.199981689453125]))
    y_offset = float(rng.choice([0.4000015258789, 0.7999954223633, 0.0]))

    def fmt(pt):
        return (
            float(round(float(pt[0]))) + x_offset,
            float(round(float(pt[1]))) + y_offset,
        )

    t0 = float(rng.uniform(1000.0, 1_000_000.0))
    t = 0.0
    events = []

    start_xy = fmt(jittered[0])
    events.append(
        {
            "x": start_xy[0],
            "y": start_xy[1],
            "timestamp": t0 + t,
            "event_type": "mousedown",
            "inside_tunnel": True,
        }
    )

    last_t = -1.0  # tracks last mousemove timestamp (relative to t0)

    # Some humans emit an instantaneous first mousemove at the press point
    # before the real motion starts; replicate that occasionally.
    if rng.random() < 0.45:
        t += float(rng.uniform(0.1, 1.2))
        events.append(
            {
                "x": start_xy[0],
                "y": start_xy[1],
                "timestamp": t0 + t,
                "event_type": "mousemove",
                "inside_tunnel": True,
            }
        )
        last_t = t
        t += float(rng.uniform(70.0, 160.0))
    else:
        t += float(rng.uniform(90.0, 200.0))

    arc = 0.0
    max_events = 1500

    while arc < total_len - 1e-3 and len(events) < max_events:
        r = float(rng.random())
        if r < 0.012:
            dt = float(rng.uniform(20.0, 60.0))
        elif r < 0.05:
            dt = float(rng.uniform(12.0, 18.0))
        else:
            dt = max(2.0, float(rng.normal(8.3, 1.0)))

        idx = int(np.searchsorted(cumlen, arc))
        if idx >= n:
            idx = n - 1
        s = float(speed[idx])

        arc = min(arc + s * dt, total_len)
        t += dt

        idx2 = int(np.searchsorted(cumlen, arc))
        if idx2 >= n:
            idx2 = n - 1
        if idx2 == 0:
            pt = jittered[0]
        else:
            l0 = cumlen[idx2 - 1]
            l1 = cumlen[idx2]
            alpha = 0.0 if l1 <= l0 else (arc - l0) / (l1 - l0)
            pt = (1.0 - alpha) * jittered[idx2 - 1] + alpha * jittered[idx2]

        xy = fmt(pt)
        if t <= last_t:
            t = last_t + 0.1

        events.append(
            {
                "x": xy[0],
                "y": xy[1],
                "timestamp": t0 + t,
                "event_type": "mousemove",
                "inside_tunnel": True,
            }
        )
        last_t = t

    end_xy = fmt(jittered[-1])
    if events[-1]["x"] != end_xy[0] or events[-1]["y"] != end_xy[1]:
        t = max(t + float(rng.normal(8.3, 1.0)), last_t + 0.5)
        events.append(
            {
                "x": end_xy[0],
                "y": end_xy[1],
                "timestamp": t0 + t,
                "event_type": "mousemove",
                "inside_tunnel": True,
            }
        )
        last_t = t

    if rng.random() < 0.5:
        up_t = last_t + float(rng.uniform(0.05, 0.5))
    else:
        up_t = last_t
    events.append(
        {
            "x": end_xy[0],
            "y": end_xy[1],
            "timestamp": t0 + up_t,
            "event_type": "mouseup",
            "inside_tunnel": True,
        }
    )

    return events
