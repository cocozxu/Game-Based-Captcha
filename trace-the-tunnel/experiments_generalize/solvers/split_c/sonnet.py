"""Human-like trace generator for the trace-the-tunnel CAPTCHA (split_c / sonnet).

The tunnel is 4 chained cubic Bezier segments (16 control-points total,
matching game.js NUM_SEGMENTS=4). TUNNEL_WIDTH=38 is the half-width from
the centerline; isInsideTunnel checks dist <= 38.

Motion model:
  * Dense Bezier arc-length sampling → per-sample tangent + curvature.
  * Speed profile: base speed scaled down at curves, smoothed.
  * Tremor overlay: two sinusoidal components in the 8–12 Hz band
    (matching the classifier's tremor_power_8_12hz feature) added
    perpendicular to the tangent.
  * Low-frequency lateral wobble (2–5 Hz) on top of tremor.
  * Per-step micro-jitter (Gaussian ~0.4 px) for derivative variety.
  * ~8.33 ms inter-event dt (120 Hz) with tiny normal noise;
    occasional 2× steps (~1.5 % of events) from missed-frame.
  * 25–90 ms corner hesitations at high-curvature peaks.
  * x / y quantized to int + 0.199981689453125 / int + 0.4000015258789
    (constant canvas offset present in all human demo events).
"""

import numpy as np

X_FRAC = 0.199981689453125
Y_FRAC = 0.4000015258789

DT_MEAN = 8.33      # ms — 120 Hz display
DT_STD  = 0.55


def _bezier_pt(t, p0, p1, p2, p3):
    u = 1.0 - t
    return u*u*u*p0 + 3*u*u*t*p1 + 3*u*t*t*p2 + t*t*t*p3


def _bezier_d1(t, p0, p1, p2, p3):
    u = 1.0 - t
    return 3*u*u*(p1-p0) + 6*u*t*(p2-p1) + 3*t*t*(p3-p2)


def _bezier_d2(t, p0, p1, p2, p3):
    u = 1.0 - t
    return 6*u*(p2 - 2*p1 + p0) + 6*t*(p3 - 2*p2 + p1)


def _smooth(arr, k):
    if k <= 1 or len(arr) < 2:
        return arr.copy()
    k = min(k, len(arr))
    kern = np.ones(k) / k
    pad  = np.concatenate([arr[:k][::-1], arr, arr[-k:][::-1]])
    return np.convolve(pad, kern, mode='same')[k:k+len(arr)]


def _build_path(cps, n_per_seg=250):
    """Dense Bezier path → (pts, tangents, curvatures, arc_lengths, seg_lens)."""
    n_seg = len(cps) // 4
    pts, tangs, curvs = [], [], []
    for si in range(n_seg):
        p0, p1, p2, p3 = cps[si*4], cps[si*4+1], cps[si*4+2], cps[si*4+3]
        start_k = 0 if si == 0 else 1
        for k in range(start_k, n_per_seg + 1):
            t  = k / n_per_seg
            pt = _bezier_pt(t, p0, p1, p2, p3)
            d1 = _bezier_d1(t, p0, p1, p2, p3)
            d2 = _bezier_d2(t, p0, p1, p2, p3)
            sp = float(np.linalg.norm(d1))
            tang = d1 / (sp + 1e-9)
            cross = d1[0]*d2[1] - d1[1]*d2[0]
            curv  = abs(cross) / (sp**3 + 1e-9)
            pts.append(pt); tangs.append(tang); curvs.append(curv)
    pts   = np.asarray(pts)
    tangs = np.asarray(tangs)
    curvs = np.asarray(curvs)
    diffs    = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arc      = np.concatenate([[0.0], np.cumsum(seg_lens)])
    return pts, tangs, curvs, arc, seg_lens


def _quantize(x, y):
    return float(int(round(x)) + X_FRAC), float(int(round(y)) + Y_FRAC)


def generate(tunnel_spec: dict, seed: int) -> list[dict]:
    """Return a human-like event stream for the given tunnel."""
    s_hash = (int(seed) * 6364136223846793005 + int(tunnel_spec['tunnel_seed'])) & 0xFFFFFFFF
    rng = np.random.default_rng(s_hash)

    cps = np.array([[p['x'], p['y']] for p in tunnel_spec['control_points']], dtype=np.float64)
    half_w = float(tunnel_spec['tunnel_width'])  # 38 — half-width from centerline

    pts, tangs, curvs, arc, seg_lens = _build_path(cps)
    n_path    = len(pts)
    total_arc = float(arc[-1])

    curvs_s = _smooth(curvs, 15)

    # --- per-trace kinematic persona ---
    v_base = float(rng.uniform(550.0, 860.0))   # px/s  (humans ~500-730 on 930px arc)
    alpha  = float(rng.uniform(12.0, 30.0))     # curvature slow-down factor
    speeds = v_base / (1.0 + alpha * curvs_s)
    speeds = np.maximum(speeds, 80.0)
    speeds = _smooth(speeds, 9)

    # --- tremor: two sinusoids in the 8–12 Hz band, lateral ---
    f_trem1 = rng.uniform(8.0, 10.0)
    f_trem2 = rng.uniform(10.0, 12.0)
    a_trem1 = rng.uniform(0.3, 0.9)
    a_trem2 = rng.uniform(0.15, 0.55)
    ph1 = rng.uniform(0.0, 2*np.pi)
    ph2 = rng.uniform(0.0, 2*np.pi)

    # --- low-freq wobble (2–5 Hz): expressed over arc length ---
    n_wob = 3
    wf  = rng.uniform(0.0015, 0.008,  n_wob)  # cycles per px of arc
    wa  = rng.uniform(0.3,    1.5,    n_wob)
    wp  = rng.uniform(0.0, 2*np.pi, n_wob)
    safe_w = max(2.5, half_w - 5.5)

    events  = []
    t_ms    = float(rng.uniform(3.0e3, 6.5e5))
    s_cur   = 0.0
    t_local = 0.0   # elapsed ms since mousedown (used for tremor frequency)
    last_corner  = -1000
    n_pauses     = 0       # limit total corner pauses to ≤2 per trace
    max_iters    = 6000
    iters        = 0

    # --- mousedown ---
    off = rng.normal(0.0, 1.5, 2)
    if np.hypot(off[0], off[1]) > 5.5:
        off = off / np.hypot(off[0], off[1]) * 5.5
    qx, qy = _quantize(pts[0][0] + off[0], pts[0][1] + off[1])
    events.append({'x': qx, 'y': qy, 'timestamp': t_ms,
                   'event_type': 'mousedown', 'inside_tunnel': True})
    last_qx, last_qy = qx, qy

    # First mousemove: same coords, short delay
    first_dt = float(rng.uniform(4.0, 55.0) if rng.random() < 0.3 else rng.uniform(2.0, 12.0))
    t_ms    += first_dt
    t_local += first_dt
    events.append({'x': qx, 'y': qy, 'timestamp': t_ms,
                   'event_type': 'mousemove', 'inside_tunnel': True})

    # --- main march ---
    while s_cur < total_arc - 1.5 and iters < max_iters:
        iters += 1

        # interpolate between path samples
        idx = int(np.searchsorted(arc, s_cur) - 1)
        idx = max(0, min(idx, n_path - 2))
        frac = min(1.0, max(0.0, (s_cur - arc[idx]) / max(float(seg_lens[idx]), 1e-9)))
        pt   = pts[idx]   * (1.0 - frac) + pts[idx+1]   * frac
        tang = tangs[idx]
        nrm  = np.array([-tang[1], tang[0]])

        # lateral displacement = wobble + tremor + micro-jitter
        wobble = float(np.sum(wa * np.sin(2*np.pi * wf * s_cur + wp)))
        tremor = (a_trem1 * np.sin(2*np.pi * f_trem1 * t_local / 1000.0 + ph1)
                + a_trem2 * np.sin(2*np.pi * f_trem2 * t_local / 1000.0 + ph2))
        micro  = float(rng.normal(0.0, 0.40))
        lat    = wobble + tremor + micro
        if abs(lat) > safe_w:
            lat = np.sign(lat) * safe_w
        pt_w = pt + nrm * lat

        # timing: ~120 Hz with occasional missed-frame double-step
        if rng.random() < 0.015:
            dt = max(3.0, 2*DT_MEAN + float(rng.normal(0.0, DT_STD)))
        else:
            dt = max(2.5, DT_MEAN + float(rng.normal(0.0, DT_STD)))
        t_ms    += dt
        t_local += dt
        s_cur   += speeds[idx] * dt / 1000.0

        qx, qy = _quantize(pt_w[0], pt_w[1])
        if qx == last_qx and qy == last_qy:
            # emit ~40 % of duplicate-coord events (humans stutter in slow zones)
            if rng.random() < 0.60:
                continue
        events.append({'x': qx, 'y': qy, 'timestamp': t_ms,
                       'event_type': 'mousemove', 'inside_tunnel': True})
        last_qx, last_qy = qx, qy

        # corner hesitation: humans briefly pause at sharp bends (≤2 per trace)
        if (n_pauses < 2
                and curvs_s[idx] > 0.040
                and (idx - last_corner) > 100
                and rng.random() < 0.20):
            last_corner = idx
            n_pauses   += 1
            pause = float(rng.uniform(10.0, 30.0))
            t_ms    += pause
            t_local += pause

    # --- final approach + mouseup ---
    end_pt  = pts[-1]
    approach = tangs[-1]
    back = float(rng.uniform(2.0, 9.0))
    ex   = end_pt[0] - approach[0]*back + float(rng.normal(0.0, 1.2))
    ey   = end_pt[1] - approach[1]*back + float(rng.normal(0.0, 1.2))
    while (ex - end_pt[0])**2 + (ey - end_pt[1])**2 > 144.0 and back > 0.5:
        back *= 0.75
        ex = end_pt[0] - approach[0]*back
        ey = end_pt[1] - approach[1]*back

    dt = max(2.5, DT_MEAN + float(rng.normal(0.0, DT_STD)))
    t_ms += dt
    qx, qy = _quantize(ex, ey)
    if qx != last_qx or qy != last_qy:
        events.append({'x': qx, 'y': qy, 'timestamp': t_ms,
                       'event_type': 'mousemove', 'inside_tunnel': True})
        last_qx, last_qy = qx, qy

    events.append({'x': last_qx, 'y': last_qy,
                   'timestamp': t_ms + float(rng.uniform(0.0, 0.5)),
                   'event_type': 'mouseup', 'inside_tunnel': True})

    # strict monotonic timestamps safety pass
    for i in range(1, len(events)):
        if events[i]['timestamp'] <= events[i-1]['timestamp']:
            events[i]['timestamp'] = events[i-1]['timestamp'] + 0.05

    return events
