"""Human-like trace generator for the trace-the-tunnel CAPTCHA (split_c).

The tunnel centerline is reconstructed as 4 chained cubic Bezier segments
(16 control points: each segment = 4 consecutive CPs, with C1 continuity
across joins). This matches game.js exactly (`NUM_SEGMENTS = 4`, cubic
Bezier construction).

Motion model:
  * Dense arc-length sampling of the centerline with per-sample tangent
    and (smoothed) curvature.
  * Speed profile v(s) = v_base / (1 + alpha * curvature), bounded
    below, giving humans' characteristic corner slowdown.
  * March along arc length at ~120 Hz (dt ~ N(8.33, 0.50) ms) which
    matches every demo trace's polling rate.
  * Lateral wobble = sum of a few low-frequency sinusoids in arc-length
    + per-step micro-jitter, clipped to a safe band inside the
    tunnel width.
  * 30 - 80 ms hesitation pauses at high-curvature regions (humans
    routinely pause at corners — visible in the demo traces).
  * Start within ~6 px of the true start point; end within the end-dot
    hit radius (so game.js auto-completes the trace).
  * x / y quantized to `int + 0.199981689453125` / `int + 0.4000015258789`
    — the fractional offsets observed on every demo event (a constant
    introduced by canvas rect.left / scaling).
"""

import numpy as np


X_FRAC = 0.199981689453125
Y_FRAC = 0.4000015258789


def _bezier_pt(t, p0, p1, p2, p3):
    u = 1.0 - t
    return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3


def _bezier_d1(t, p0, p1, p2, p3):
    u = 1.0 - t
    return 3 * u * u * (p1 - p0) + 6 * u * t * (p2 - p1) + 3 * t * t * (p3 - p2)


def _bezier_d2(t, p0, p1, p2, p3):
    u = 1.0 - t
    return 6 * u * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1)


def _quantize(x, y):
    return float(int(round(x)) + X_FRAC), float(int(round(y)) + Y_FRAC)


def _smooth(arr, k):
    if k <= 1 or len(arr) <= 1:
        return arr.copy()
    n = len(arr)
    kk = min(k, n)
    kern = np.ones(kk) / kk
    pad = np.concatenate([arr[:kk][::-1], arr, arr[-kk:][::-1]])
    out = np.convolve(pad, kern, mode="same")
    return out[kk:kk + n]


def _build_path(cps, n_per_seg=200):
    """Returns (pts, tangents, curvatures, arc_lengths, segment_lengths)."""
    n_seg = len(cps) // 4
    pts, tangs, curvs = [], [], []
    for si in range(n_seg):
        p0 = cps[si * 4]
        p1 = cps[si * 4 + 1]
        p2 = cps[si * 4 + 2]
        p3 = cps[si * 4 + 3]
        start_k = 0 if si == 0 else 1
        for k in range(start_k, n_per_seg + 1):
            t = k / n_per_seg
            pt = _bezier_pt(t, p0, p1, p2, p3)
            d1 = _bezier_d1(t, p0, p1, p2, p3)
            d2 = _bezier_d2(t, p0, p1, p2, p3)
            sp = float(np.linalg.norm(d1))
            tang = d1 / (sp + 1e-9)
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            curv = abs(cross) / (sp ** 3 + 1e-9)
            pts.append(pt)
            tangs.append(tang)
            curvs.append(curv)
    pts = np.asarray(pts)
    tangs = np.asarray(tangs)
    curvs = np.asarray(curvs)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    s_arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    return pts, tangs, curvs, s_arc, seg_lens


def generate(tunnel_spec, seed):
    s_hash = (int(seed) * 2654435761 + int(tunnel_spec["tunnel_seed"])) & 0xFFFFFFFF
    rng = np.random.default_rng(s_hash)

    cps = np.array(
        [[p["x"], p["y"]] for p in tunnel_spec["control_points"]],
        dtype=np.float64,
    )
    # game.js: isInsideTunnel(...) => distToCenterline(...) <= TUNNEL_WIDTH (= 38).
    # So tunnel_width is the actual half-width allowed.
    width = float(tunnel_spec["tunnel_width"])

    pts, tangs, curvs, s_arc, seg_lens = _build_path(cps)
    n_path = len(pts)
    total_len = float(s_arc[-1])

    curvs_s = _smooth(curvs, 11)

    # Per-trace persona — speeds tuned to give 1.3 - 2.2 s durations on
    # the ~950 px centerlines, matching demo human trace durations.
    v_base = float(rng.uniform(520.0, 760.0))      # px / s
    alpha = float(rng.uniform(10.0, 22.0))         # curvature slowdown
    speeds = v_base / (1.0 + alpha * curvs_s)
    speeds = np.maximum(speeds, 90.0)
    speeds = _smooth(speeds, 7)

    # Lateral wobble: small sum of low-frequency sinusoids over arc length,
    # plus per-step Gaussian micro-jitter at emit time.
    n_modes = 3
    wob_freq = rng.uniform(0.002, 0.012, n_modes)
    wob_amp = rng.uniform(0.35, 1.4, n_modes)
    wob_phase = rng.uniform(0.0, 2.0 * np.pi, n_modes)
    safe_w = max(2.0, width - 6.0)

    def lateral_at(s):
        v = float(np.sum(wob_amp * np.sin(2.0 * np.pi * wob_freq * s + wob_phase)))
        return v

    DT_MEAN = 8.33
    DT_STD = 0.50

    events = []
    t_ms = float(rng.uniform(2.0e4, 6.0e5))

    # ---- mousedown ------------------------------------------------------
    start_off = rng.normal(0.0, 1.8, 2)
    d0 = float(np.hypot(start_off[0], start_off[1]))
    if d0 > 6.0:
        start_off = start_off * (6.0 / d0)
    sx = pts[0][0] + start_off[0]
    sy = pts[0][1] + start_off[1]
    qx, qy = _quantize(sx, sy)
    events.append({
        "x": qx, "y": qy, "timestamp": t_ms,
        "event_type": "mousedown", "inside_tunnel": True,
    })
    last_qx, last_qy = qx, qy

    # First mousemove typically at (or 1px from) mousedown, after a small delay
    if rng.random() < 0.28:
        first_delay = float(rng.uniform(30.0, 70.0))
    else:
        first_delay = float(rng.uniform(3.0, 14.0))
    t_ms += first_delay
    events.append({
        "x": qx, "y": qy, "timestamp": t_ms,
        "event_type": "mousemove", "inside_tunnel": True,
    })

    # ---- main march -----------------------------------------------------
    s_cur = 0.0
    last_corner_idx = -1000
    max_iters = 5000
    iters = 0

    while s_cur < total_len - 1.5 and iters < max_iters:
        iters += 1
        idx = int(np.searchsorted(s_arc, s_cur) - 1)
        idx = max(0, min(idx, n_path - 2))
        local_seg = max(float(seg_lens[idx]), 1e-9)
        frac = min(1.0, max(0.0, (s_cur - s_arc[idx]) / local_seg))

        pt = pts[idx] * (1.0 - frac) + pts[idx + 1] * frac
        tang = tangs[idx]
        normal = np.array([-tang[1], tang[0]])

        wobble = lateral_at(s_cur) + float(rng.normal(0.0, 0.35))
        if abs(wobble) > safe_w:
            wobble = np.sign(wobble) * safe_w
        pt_w = pt + normal * wobble

        dt = max(2.5, DT_MEAN + float(rng.normal(0.0, DT_STD)))
        t_ms += dt
        speed = float(speeds[idx])
        s_cur += speed * dt / 1000.0

        qx, qy = _quantize(pt_w[0], pt_w[1])
        if qx == last_qx and qy == last_qy:
            # Most often skip duplicates; sometimes emit one to mimic
            # the "stutter" the demo traces show during slow zones.
            if rng.random() < 0.6:
                continue
        events.append({
            "x": qx, "y": qy, "timestamp": t_ms,
            "event_type": "mousemove", "inside_tunnel": True,
        })
        last_qx, last_qy = qx, qy

        # Corner hesitation pause
        if (curvs_s[idx] > 0.035
                and (idx - last_corner_idx) > 45
                and rng.random() < 0.40):
            last_corner_idx = idx
            t_ms += float(rng.uniform(25.0, 85.0))

    # ---- mouseup --------------------------------------------------------
    # Land within the end-dot hit radius (14 px + 4 tolerance = 18 px).
    # Use the path tangent so the approach direction looks natural.
    end_pt = pts[-1]
    approach_dir = tangs[-1]
    back_off = float(rng.uniform(3.0, 10.0))
    end_x = end_pt[0] - approach_dir[0] * back_off + float(rng.normal(0.0, 1.5))
    end_y = end_pt[1] - approach_dir[1] * back_off + float(rng.normal(0.0, 1.5))
    # Ensure well within hit radius (target <= 12 px from end)
    while (end_x - end_pt[0]) ** 2 + (end_y - end_pt[1]) ** 2 > 144.0 and back_off > 0.5:
        back_off *= 0.7
        end_x = end_pt[0] - approach_dir[0] * back_off
        end_y = end_pt[1] - approach_dir[1] * back_off

    dt = max(2.5, DT_MEAN + float(rng.normal(0.0, DT_STD)))
    t_ms += dt
    qx, qy = _quantize(end_x, end_y)
    if qx != last_qx or qy != last_qy:
        events.append({
            "x": qx, "y": qy, "timestamp": t_ms,
            "event_type": "mousemove", "inside_tunnel": True,
        })
        last_qx, last_qy = qx, qy

    mu_offset = float(rng.uniform(0.0, 0.4))
    events.append({
        "x": last_qx, "y": last_qy,
        "timestamp": t_ms + mu_offset,
        "event_type": "mouseup", "inside_tunnel": True,
    })

    # Enforce strict monotonicity on timestamps as a safety net.
    for i in range(1, len(events)):
        if events[i]["timestamp"] <= events[i - 1]["timestamp"]:
            events[i]["timestamp"] = events[i - 1]["timestamp"] + 0.05

    return events
