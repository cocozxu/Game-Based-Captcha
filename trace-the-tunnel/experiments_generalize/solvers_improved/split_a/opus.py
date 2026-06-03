"""Improved opus solver for trace-the-tunnel — split_a.

Three structural fixes versus solvers/split_a/opus.py:

1. Elastic-band path (not centerline-following).
   Humans treat the tunnel as a corridor and find the smoothest path
   through it — they don't track the centerline at all, they just stay
   inside. We mimic this by iteratively Laplacian-smoothing the
   centerline while constraining each point to stay within the safe
   band. Like pulling an elastic band taut inside a curved corridor,
   the path relaxes toward chords through gentle bends and only bends
   where the walls force it to.

2. Submovement-based speed (not a single v(s) curve).
   Measured: human speed_mean=0.49, speed_max=1.35 (max/mean=2.77),
   skew=+0.87. Old opus capped at v_max=0.6 with skew~0 — no peaks.
   Here position-along-arc is the sum of 6-10 overlapping
   minimum-jerk submovements; velocity is the sum of bell curves and
   naturally produces right-skewed speed with multiple peaks.

3. AR(1) lateral tremor, not pure 8-12Hz sinusoid (avoids
   tremor_power_8_12hz lighting up).
"""

import math
import numpy as np

X_OFFSET = 0.199981689453125
Y_OFFSET = 0.400001525878906


# --- geometry helpers --------------------------------------------------------

def _bezier_seg(p0, p1, p2, p3, n, include_end):
    m = n + (1 if include_end else 0)
    ts = np.arange(m) / n
    u = 1.0 - ts
    x = u**3 * p0[0] + 3 * u**2 * ts * p1[0] + 3 * u * ts**2 * p2[0] + ts**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * ts * p1[1] + 3 * u * ts**2 * p2[1] + ts**3 * p3[1]
    return np.stack([x, y], axis=1)


def _build_centerline(control_points, n_per_seg=400):
    pts = [(cp['x'], cp['y']) if isinstance(cp, dict) else (cp[0], cp[1])
           for cp in control_points]
    n_segs = len(pts) // 4
    if n_segs == 0:
        return np.array(pts, dtype=float)
    chunks = []
    for s_idx in range(n_segs):
        i = s_idx * 4
        seg = pts[i:i + 4]
        last = (s_idx == n_segs - 1)
        chunks.append(_bezier_seg(seg[0], seg[1], seg[2], seg[3], n_per_seg, last))
    return np.concatenate(chunks, axis=0)


def _arc_length(path):
    diffs = np.diff(path, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _tangents_and_signed_curvature(path):
    n = len(path)
    tangents = np.zeros_like(path)
    tangents[1:-1] = path[2:] - path[:-2]
    tangents[0] = path[1] - path[0]
    tangents[-1] = path[-1] - path[-2]
    tn = np.linalg.norm(tangents, axis=1, keepdims=True)
    tn[tn < 1e-9] = 1.0
    tangents = tangents / tn

    kappa = np.zeros(n)
    v1 = path[1:-1] - path[:-2]
    v2 = path[2:] - path[1:-1]
    l1 = np.linalg.norm(v1, axis=1)
    l2 = np.linalg.norm(v2, axis=1)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    denom = np.maximum(l1 * l2, 1e-9)
    sin_t = np.clip(cross / denom, -1.0, 1.0)
    kappa[1:-1] = np.arcsin(sin_t) / np.maximum(l1, 1e-9)
    return tangents, kappa


def _smooth(arr, win):
    if win <= 1 or len(arr) < 3:
        return arr.copy()
    win = min(win, len(arr) // 2)
    if win <= 1:
        return arr.copy()
    pad = np.pad(arr, win, mode='edge')
    kernel = np.ones(2 * win + 1) / (2 * win + 1)
    return np.convolve(pad, kernel, mode='same')[win:-win]


def _smooth_2d(path, win):
    if win <= 1 or len(path) < 3:
        return path.copy()
    return np.stack([_smooth(path[:, 0], win), _smooth(path[:, 1], win)], axis=1)


def _interp_arr(path, arc, s_vals):
    s_vals = np.clip(s_vals, 0.0, arc[-1])
    idx = np.searchsorted(arc, s_vals, side='right')
    idx = np.clip(idx, 1, len(arc) - 1)
    s0 = arc[idx - 1]
    s1 = arc[idx]
    f = ((s_vals - s0) / np.maximum(s1 - s0, 1e-9))[:, None]
    return path[idx - 1] + (path[idx] - path[idx - 1]) * f


def _elastic_band(centerline, half_w, win, n_pass):
    """Find the smoothest path through the tunnel — like pulling an elastic
    band taut inside a curved corridor.

    Each pass: (a) box-smooth the whole path with a wide window (shortens
    the path toward straight lines / chords across gentle bends), (b)
    project any point that has drifted past the safe band back to the
    boundary, by snapping toward its nearest centerline point. After 2-3
    passes the path settles into something close to the shortest route
    the walls allow: chord-cuts where it can, hugging the boundary only
    where it must. This is what humans do — pick the easiest line, don't
    track the centerline.

    The 1-step Laplacian / 3-tap kernel is far too gentle here: with
    ~1600 dense path points the local neighborhood is essentially
    straight, so each iteration barely moves anything. A wide box filter
    (win ~200 = ½ of one bezier segment) attenuates the high-frequency
    oscillations of the control polygon in a single shot.
    """
    safe = float(half_w) - 4.0
    p = centerline.copy()
    n = len(p)
    step = max(1, n // 240)
    cl_check = centerline[::step]

    for _ in range(n_pass):
        smoothed = _smooth_2d(p, win)
        smoothed[0] = centerline[0]
        smoothed[-1] = centerline[-1]

        diffs = smoothed[:, None, :] - cl_check[None, :, :]
        d2 = (diffs ** 2).sum(axis=2)
        j_min = d2.argmin(axis=1)
        d_min = np.sqrt(d2[np.arange(n), j_min])
        mask = d_min > safe
        if mask.any():
            nearest = cl_check[j_min[mask]]
            direction = smoothed[mask] - nearest
            smoothed[mask] = nearest + direction * (safe / d_min[mask])[:, None]
            smoothed[0] = centerline[0]
            smoothed[-1] = centerline[-1]
        p = smoothed

    return p


# --- main --------------------------------------------------------------------

def generate(tunnel_spec, seed):
    rng = np.random.default_rng(int(seed))

    cps = tunnel_spec['control_points']
    # game.js: distToCenterline <= TUNNEL_WIDTH (=38). So tunnel_width is the
    # half-width allowed — the path may sit up to ~38 px from centerline.
    half_w = float(tunnel_spec['tunnel_width'])

    path = _build_centerline(cps, n_per_seg=400)
    n_pts = len(path)

    # --- Elastic-band lazy path ---------------------------------------------
    # Humans don't track the centerline — they just stay in the tunnel and
    # take the smoothest line they can. Wide box-smooth the centerline to
    # collapse the control polygon's high-frequency oscillations into a
    # chord-like path, then project any points outside the safe band back
    # to the boundary. 2-3 passes typically converges; n_pass and win
    # vary per-trace so different attempts settle on slightly different
    # paths through the same tunnel.
    laziness_win = int(rng.integers(180, 260))
    n_pass = int(rng.integers(2, 5))
    lazy_path = _elastic_band(path, half_w, win=laziness_win, n_pass=n_pass)

    # Slow lateral undulation + per-trace global bias — kept small because
    # the elastic-band path already carries most of the lateral deviation
    # via chord-cutting, and humans look fairly steady once committed to a
    # line. Apply along the lazy path's own normals (not the centerline's,
    # which still wiggle at high frequency).
    lazy_arc = _arc_length(lazy_path)
    lazy_tangents, _ = _tangents_and_signed_curvature(lazy_path)
    lazy_normals = np.stack([-lazy_tangents[:, 1], lazy_tangents[:, 0]], axis=1)

    bias = float(rng.normal(0.0, 1.8))
    n_modes = 3
    wob_freq = rng.uniform(0.0015, 0.006, n_modes)
    wob_amp = rng.uniform(0.4, 1.5, n_modes)
    wob_phase = rng.uniform(0.0, 2 * math.pi, n_modes)
    wob = np.zeros(n_pts)
    for f, a, ph in zip(wob_freq, wob_amp, wob_phase):
        wob += a * np.sin(2 * math.pi * f * lazy_arc + ph)
    lateral = bias + wob
    safe = max(2.0, half_w - 5.0)
    lateral = np.clip(lateral, -safe, safe)
    n_blend = 60
    edge_fade = np.minimum(np.arange(n_pts), np.arange(n_pts)[::-1])
    fade = np.minimum(1.0, edge_fade / max(n_blend, 1))
    lateral = lateral * fade
    lazy_path = lazy_path + lazy_normals * lateral[:, None]
    lazy_path[0] = path[0]
    lazy_path[-1] = path[-1]

    # Final safety pass at full centerline resolution — adding the wobble
    # may push a few points past the safe band. Clamp using true MIN-distance
    # to the centerline (the metric game.js actually checks).
    for i in range(n_pts):
        dists = np.sqrt((path[:, 0] - lazy_path[i, 0]) ** 2
                        + (path[:, 1] - lazy_path[i, 1]) ** 2)
        d_min = dists.min()
        if d_min > safe:
            j = int(dists.argmin())
            direction = lazy_path[i] - path[j]
            lazy_path[i] = path[j] + direction * (safe / d_min)

    lazy_arc = _arc_length(lazy_path)
    lazy_len = float(lazy_arc[-1])

    # --- Submovement schedule -----------------------------------------------
    # Heavy-tailed chunk size distribution + duration NOT scaled with chunk
    # size → big chunks get fast peaks while small chunks crawl. This gives
    # the right-skewed speed distribution humans exhibit (skew ~+0.9).
    n_sub = int(rng.integers(8, 13))
    T_total = float(rng.uniform(1100.0, 1550.0))

    # Heavy-tailed chunk sizes: most submovements small, 1-2 outsized to
    # produce the sharp peaks (humans speed_max ~1.35, peak/mean ~2.8).
    chunk_weights = np.exp(rng.normal(0.0, 0.55, n_sub))
    chunk_weights *= lazy_len / chunk_weights.sum()
    edges = np.concatenate([[0.0], np.cumsum(chunk_weights)])
    edges[-1] = lazy_len
    sub_ds = np.diff(edges)

    # All submovements get roughly the same duration → larger chunks ⇒
    # higher peak speed (peak ∝ chunk / duration in min-jerk).
    overlap = float(rng.uniform(0.25, 0.40))
    nominal_step = T_total / n_sub
    base_dur = nominal_step / (1.0 - overlap)
    durations = base_dur * np.exp(rng.normal(0.0, 0.10, n_sub))
    durations = np.clip(durations, base_dur * 0.65, base_dur * 1.4)

    starts = np.concatenate([[0.0], np.cumsum(np.full(n_sub - 1, nominal_step))])
    ends = starts + durations
    T_actual = float(ends.max())

    # --- Time samples at ~120 Hz. Per-trace dt persona varies across the
    # measured human ranges (dt_min 7.0-7.5, dt_max 9.4-24.4 ms) so the
    # attack distribution isn't a tight point in dt-feature space.
    dt_mean_pers = float(rng.normal(8.36, 0.06))
    dt_std_pers = float(rng.uniform(0.25, 0.95))
    dt_floor = float(rng.uniform(6.6, 7.6))
    dt_ceil = float(rng.uniform(10.5, 18.0))
    times = [0.0]
    while times[-1] < T_actual - 0.1:
        dt = float(rng.normal(dt_mean_pers, dt_std_pers))
        dt = max(dt_floor, min(dt_ceil, dt))
        times.append(times[-1] + dt)
    times = np.array(times)

    # --- s(t) = sum of minimum-jerk submovement contributions --------------
    s_t = np.zeros_like(times)
    for i in range(n_sub):
        tau = np.clip((times - starts[i]) / durations[i], 0.0, 1.0)
        mj = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        s_t += sub_ds[i] * mj
    s_t = np.minimum(s_t, lazy_len)

    positions = _interp_arr(lazy_path, lazy_arc, s_t)
    # High-frequency lateral tremor — bursty, not pure 8–12 Hz sinusoid
    # (which would over-light up tremor_power_8_12hz). Measured: human
    # accel_std ~0.019, jerk_std ~0.0039 — min-jerk alone is ~30% below.
    # AR(1) noise with α=0.55 gives jerky-but-not-periodic perturbation.
    n_evt = len(times)
    noise = np.zeros((n_evt, 2))
    alpha_tr = 0.55
    sigma_tr = 0.40
    for i in range(1, n_evt):
        noise[i] = alpha_tr * noise[i - 1] + rng.normal(0.0, sigma_tr, 2)
    positions = positions + noise

    # AR(1) drift can accumulate to ~15-20 px over a full trace when the
    # chord-cut path is already sitting near the boundary, occasionally
    # pushing events outside. Final clip per-event using min-distance to
    # centerline so every emitted point lands inside the tunnel.
    clip_margin = max(2.0, float(half_w) - 2.0)
    cl_step = max(1, n_pts // 400)
    cl_clip = path[::cl_step]
    diffs_p = positions[:, None, :] - cl_clip[None, :, :]
    d2_p = (diffs_p ** 2).sum(axis=2)
    j_p = d2_p.argmin(axis=1)
    d_p = np.sqrt(d2_p[np.arange(n_evt), j_p])
    mask_p = d_p > clip_margin
    if mask_p.any():
        near = cl_clip[j_p[mask_p]]
        direction = positions[mask_p] - near
        positions[mask_p] = near + direction * (clip_margin / d_p[mask_p])[:, None]

    # --- Build event list ---------------------------------------------------
    t0 = float(rng.uniform(100000.0, 600000.0))
    events = []

    p_start = lazy_path[0]
    x_start = round(float(p_start[0])) + X_OFFSET
    y_start = round(float(p_start[1])) + Y_OFFSET

    events.append({
        "x": x_start, "y": y_start,
        "timestamp": t0,
        "event_type": "mousedown",
        "inside_tunnel": True,
    })

    r = rng.random()
    if r < 0.18:
        first_gap = float(rng.uniform(0.3, 4.5))
    elif r < 0.92:
        first_gap = float(rng.uniform(7.0, 22.0))
    else:
        first_gap = float(rng.uniform(45.0, 180.0))
    t_press = t0 + first_gap
    events.append({
        "x": x_start, "y": y_start,
        "timestamp": t_press,
        "event_type": "mousemove",
        "inside_tunnel": True,
    })

    last_qx, last_qy = x_start, y_start
    for i in range(1, len(times)):
        x = round(float(positions[i, 0])) + X_OFFSET
        y = round(float(positions[i, 1])) + Y_OFFSET
        t_event = t_press + float(times[i])
        if x == last_qx and y == last_qy and rng.random() < 0.6:
            continue
        events.append({
            "x": x, "y": y,
            "timestamp": t_event,
            "event_type": "mousemove",
            "inside_tunnel": True,
        })
        last_qx, last_qy = x, y

    # Snap to centerline endpoint pixel so game.js auto-completes.
    p_end = lazy_path[-1]
    x_end = round(float(p_end[0])) + X_OFFSET
    y_end = round(float(p_end[1])) + Y_OFFSET
    last_t = events[-1]['timestamp']
    if events[-1]['x'] != x_end or events[-1]['y'] != y_end:
        dt = max(2.5, float(rng.normal(8.33, 0.32)))
        last_t += dt
        events.append({
            "x": x_end, "y": y_end,
            "timestamp": last_t,
            "event_type": "mousemove",
            "inside_tunnel": True,
        })

    if rng.random() < 0.30:
        for _ in range(int(rng.integers(1, 4))):
            dt = max(2.5, float(rng.normal(8.33, 0.32)))
            last_t += dt
            events.append({
                "x": events[-1]['x'], "y": events[-1]['y'],
                "timestamp": last_t,
                "event_type": "mousemove",
                "inside_tunnel": True,
            })

    events.append({
        "x": events[-1]['x'], "y": events[-1]['y'],
        "timestamp": events[-1]['timestamp'],
        "event_type": "mouseup",
        "inside_tunnel": True,
    })

    for i in range(1, len(events)):
        if events[i]['timestamp'] <= events[i - 1]['timestamp']:
            events[i]['timestamp'] = events[i - 1]['timestamp'] + 0.05

    return events
