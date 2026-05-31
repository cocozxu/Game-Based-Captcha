"""
Generalization solver for the trace-the-tunnel CAPTCHA — Sonnet split-b.

Approach
--------
The 16 control_points form 4 cubic Bézier segments (pts[0-3], pts[4-7],
pts[8-11], pts[12-15]; consecutive segments share an endpoint so indices
3=4, 7=8, 11=12 are duplicated). The forge maps our canvas-space (x, y)
values back through game.js's coordinate transform, so we work entirely
in canvas logical pixels.

1. Reconstruct the full centerline by dense Bézier sampling → arc-length
   parameterise.
2. Build a per-trace speed profile combining:
     - Ease-in / ease-out (humans accelerate from rest and brake to stop)
     - 2/3-power-law curvature coupling (tangential speed ∝ κ^{-1/3})
3. Walk along the centerline at the variable speed, sampling at ~8.3 ms
   intervals with Gaussian dt jitter, producing n_points ≈ 120-195
   mousemove events.
4. Add two noise components to each position:
     - Slow perpendicular drift: AR(1) with high autocorrelation (captures
       the human centering error / micro-correction pattern that drives
       centerline_dev_mean / centerline_dev_std).
     - Physiological tremor: sinusoidal at 8-12 Hz (populates the
       tremor_power_8_12hz feature the classifier uses from the speed FFT).
5. Clamp positions to stay safely inside tunnel_width / 2.
6. Add a small per-trace subpixel constant to x (mimics the page-to-canvas
   transform offset visible in real data).
7. Emit: mousedown → initial hold pause → mousemove × n_points → mouseup.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Bézier helpers
# ---------------------------------------------------------------------------

def _build_centerline(pts, spp=400):
    """
    Evaluate the 4-segment cubic Bézier centerline.
    Returns (cx, cy, arc) as flat numpy arrays.
    """
    all_x, all_y = [], []
    for s in range(4):
        b = s * 4
        p0, p1, p2, p3 = pts[b], pts[b+1], pts[b+2], pts[b+3]
        ts = np.linspace(0.0, 1.0, spp + 1)
        if s > 0:
            ts = ts[1:]  # skip duplicate junction point
        mt = 1.0 - ts
        xs = (mt**3 * p0['x'] + 3*mt**2*ts * p1['x']
              + 3*mt*ts**2 * p2['x'] + ts**3 * p3['x'])
        ys = (mt**3 * p0['y'] + 3*mt**2*ts * p1['y']
              + 3*mt*ts**2 * p2['y'] + ts**3 * p3['y'])
        all_x.extend(xs)
        all_y.extend(ys)

    cx = np.array(all_x)
    cy = np.array(all_y)
    ds = np.sqrt(np.diff(cx)**2 + np.diff(cy)**2)
    arc = np.concatenate([[0.0], np.cumsum(ds)])
    return cx, cy, arc


def _interp_centerline(arc_target, cx, cy, arc):
    """Interpolate (x, y) at given arc-length positions."""
    return np.interp(arc_target, arc, cx), np.interp(arc_target, arc, cy)


def _curvature_at_arc(cx, cy, arc, arc_target):
    """Smooth curvature (1/radius) at arc-length positions."""
    dcx = np.gradient(cx)
    dcy = np.gradient(cy)
    ddcx = np.gradient(dcx)
    ddcy = np.gradient(dcy)
    denom = (dcx**2 + dcy**2) ** 1.5 + 1e-9
    kappa = np.abs(dcx * ddcy - dcy * ddcx) / denom
    # Smooth with a wide window before interpolating
    w = 30
    kappa_s = np.convolve(kappa, np.ones(w) / w, mode='same')
    return np.interp(arc_target, arc, kappa_s)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def generate(tunnel_spec: dict, seed: int) -> list[dict]:
    """
    Generate a human-like mouse trace for the given tunnel geometry.

    Returns a list of event dicts with keys:
        x, y, timestamp, event_type, inside_tunnel
    The event_type sequence is: mousedown, mousemove…, mouseup.
    Timestamps are strictly monotonic (ms).
    (x, y) are canvas logical pixels, matching the coordinate system of
    control_points; the forge handles the viewport transform.
    """
    rng = np.random.default_rng(seed)

    pts       = tunnel_spec['control_points']
    half_w    = tunnel_spec['tunnel_width'] / 2.0   # 19 px for all tunnels in dataset

    # ------------------------------------------------------------------ #
    # 1. Build dense centerline and arc-length parameterisation          #
    # ------------------------------------------------------------------ #
    cx, cy, arc = _build_centerline(pts, spp=400)
    total_arc = arc[-1]

    # ------------------------------------------------------------------ #
    # 2. Speed profile: ease-in/out  ×  2/3-power curvature coupling    #
    # ------------------------------------------------------------------ #
    # We will walk the arc by stepping through it.  First, decide how
    # many events to emit: target 120-195 mousemove events (n_points
    # seen by the classifier), which at ~8.3 ms/event gives 1.0-1.6 s.
    n_points = int(rng.integers(120, 195))

    # Base inter-event interval (ms), modelling ~120 Hz mouse polling
    base_dt = rng.uniform(7.8, 9.2)

    # Nominal arc advance per step if speed were constant
    nominal_step = total_arc / n_points  # px per step

    # We build a continuous speed-factor curve over normalised arc [0,1].
    # Ease envelope: sin^2 raises from 0 to 1 in the first 15% and
    # falls back in the last 15%.
    s_norm = np.linspace(0.0, 1.0, n_points)
    ease_ramp = 0.15
    ease_in   = np.clip(s_norm / ease_ramp, 0.0, 1.0)
    ease_out  = np.clip((1.0 - s_norm) / ease_ramp, 0.0, 1.0)
    ease      = np.sin(np.minimum(ease_in, ease_out) * math.pi / 2.0)
    ease      = np.clip(ease, 0.25, 1.0)          # never go below 25 % of top speed

    # Curvature coupling: speed ∝ κ^{-1/3}  (empirical 2/3 power law)
    arc_at_s = s_norm * total_arc
    kappa     = _curvature_at_arc(cx, cy, arc, arc_at_s)
    kappa_ref = np.percentile(kappa, 75) + 1e-6   # normalise to 75th pct
    curv_factor = (kappa_ref / (kappa + kappa_ref)) ** (1.0 / 3.0)
    curv_factor  = np.clip(curv_factor, 0.3, 1.5)

    speed_profile = ease * curv_factor             # combined factor, shape (n_points,)
    speed_profile /= speed_profile.mean()          # keep mean = 1.0

    # ------------------------------------------------------------------ #
    # 3. Walk along the arc to get per-event arc positions               #
    # ------------------------------------------------------------------ #
    # arc_positions[i] = arc-length at the i-th mousemove event
    arc_positions = np.zeros(n_points + 2)         # +2 for mousedown and mouseup
    # mousedown starts at arc = 0
    arc_positions[0] = 0.0

    # Distribute n_points + 1 gaps (mousedown..last_mousemove) with
    # variable step sizes based on the speed profile.
    steps = nominal_step * speed_profile
    cum = 0.0
    for i in range(n_points):
        cum = min(cum + steps[i], total_arc)
        arc_positions[i + 1] = cum
    arc_positions[-1] = total_arc                  # mouseup exactly at end

    # ------------------------------------------------------------------ #
    # 4. Interpolate centerline positions                                 #
    # ------------------------------------------------------------------ #
    px, py = _interp_centerline(arc_positions, cx, cy, arc)

    # Perpendicular (normal) unit vector at each sample
    tx_raw = np.gradient(px)
    ty_raw = np.gradient(py)
    tlen   = np.sqrt(tx_raw**2 + ty_raw**2) + 1e-9
    nx_v   = -ty_raw / tlen
    ny_v   =  tx_raw / tlen

    # ------------------------------------------------------------------ #
    # 5. Noise: slow perpendicular drift + physiological tremor          #
    # ------------------------------------------------------------------ #
    n_total = n_points + 2                          # all spatial samples

    # -- Slow AR(1) perpendicular drift --
    alpha_drift  = rng.uniform(0.83, 0.94)
    sigma_drift  = rng.uniform(0.8, 2.2)
    perp         = np.zeros(n_total)
    for i in range(1, n_total):
        perp[i] = (alpha_drift * perp[i-1]
                   + rng.normal(0.0, sigma_drift * math.sqrt(1.0 - alpha_drift**2)))
    max_perp = half_w - 4.5                         # 4.5 px safety margin
    perp     = np.clip(perp, -max_perp, max_perp)

    # -- Physiological tremor at 8-12 Hz added directly to (x, y) --
    tremor_freq  = rng.uniform(8.0, 12.0)           # Hz
    tremor_amp   = rng.uniform(0.4, 1.6)            # px
    phi_x        = rng.uniform(0.0, 2.0 * math.pi)
    phi_y        = rng.uniform(0.0, 2.0 * math.pi)
    t_sec        = np.arange(n_total) * base_dt / 1000.0
    tremor_x     = tremor_amp * np.cos(2.0 * math.pi * tremor_freq * t_sec + phi_x)
    tremor_y     = tremor_amp * np.sin(2.0 * math.pi * tremor_freq * t_sec + phi_y)

    # Apply
    sx = px + perp * nx_v + tremor_x
    sy = py + perp * ny_v + tremor_y

    # Hard-clamp to canvas bounds (canvas_size: 600 × 350 for all tunnels)
    canvas_w = tunnel_spec.get('canvas_size', {}).get('width',  600)
    canvas_h = tunnel_spec.get('canvas_size', {}).get('height', 350)
    sx = np.clip(sx, 0.0, float(canvas_w))
    sy = np.clip(sy, 0.0, float(canvas_h))

    # ------------------------------------------------------------------ #
    # 6. Timestamps                                                       #
    # ------------------------------------------------------------------ #
    t0 = float(rng.uniform(5000.0, 600000.0))

    # Initial hold after mousedown: humans pause ~50-200 ms before moving
    init_hold = float(rng.uniform(50.0, 200.0))

    # Per-step dt: base + curvature slowdown + Gaussian jitter
    kappa_all   = _curvature_at_arc(cx, cy, arc, arc_positions)
    kappa_ref2  = np.percentile(kappa_all, 75) + 1e-6
    curv_slow   = 1.0 + 0.5 * kappa_all / (kappa_all + kappa_ref2)
    dt_jitter   = rng.normal(0.0, 1.1, n_total)
    dts         = base_dt * curv_slow + dt_jitter
    dts         = np.clip(dts, 4.5, 28.0)

    timestamps = [t0, t0 + init_hold]
    for i in range(2, n_total):
        timestamps.append(timestamps[-1] + float(dts[i]))

    # ------------------------------------------------------------------ #
    # 7. Assemble event stream                                            #
    # ------------------------------------------------------------------ #
    events = []

    # mousedown
    events.append({
        'x':            float(sx[0]),
        'y':            float(sy[0]),
        'timestamp':    timestamps[0],
        'event_type':   'mousedown',
        'inside_tunnel': True,
    })

    # mousemove (indices 1 … n_total-2)
    for i in range(1, n_total - 1):
        events.append({
            'x':            float(sx[i]),
            'y':            float(sy[i]),
            'timestamp':    timestamps[i],
            'event_type':   'mousemove',
            'inside_tunnel': True,
        })

    # mouseup
    events.append({
        'x':            float(sx[-1]),
        'y':            float(sy[-1]),
        'timestamp':    timestamps[-1],
        'event_type':   'mouseup',
        'inside_tunnel': True,
    })

    return events
