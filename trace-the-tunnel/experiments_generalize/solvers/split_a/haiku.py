import numpy as np

def generate(tunnel_spec: dict, seed: int) -> list[dict]:
    """
    Generate human-like mouse traces by interpolating the tunnel centerline
    and adding realistic speed/timing variations.

    Approach:
    1. Smooth interpolation of control points using Catmull-Rom splines
    2. Adaptive point sampling based on local curvature (tight curves = more points)
    3. Natural timing with variable inter-event delays (8-20ms base, occasional pauses)
    4. Jitter to avoid perfect linearity
    """
    np.random.seed(seed)

    control_points = np.array(tunnel_spec['control_points'])
    tunnel_width = tunnel_spec['tunnel_width']
    half_width = tunnel_width / 2

    # Catmull-Rom spline interpolation for smooth centerline
    def catmull_rom(p0, p1, p2, p3, t):
        t2, t3 = t*t, t*t*t
        return 0.5 * (
            2*p1 + (-p0 + p2)*t + (2*p0 - 5*p1 + 4*p2 - p3)*t2 +
            (-p0 + 3*p1 - 3*p2 + p3)*t3
        )

    # Generate smooth centerline from control points
    centerline = []
    for i in range(len(control_points) - 1):
        p0 = control_points[max(0, i-1)]
        p1 = control_points[i]
        p2 = control_points[i+1]
        p3 = control_points[min(len(control_points)-1, i+2)]

        for t in np.linspace(0, 1, 50, endpoint=False):
            x = catmull_rom(p0[0], p1[0], p2[0], p3[0], t)
            y = catmull_rom(p0[1], p1[1], p2[1], p3[1], t)
            centerline.append([x, y])

    centerline.append(control_points[-1])
    centerline = np.array(centerline)

    # Calculate curvature for adaptive sampling
    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])
    seg_lengths = np.sqrt(dx**2 + dy**2)

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx*ddy - dy*ddx) / np.maximum(seg_lengths**2, 1e-6)

    # Sample adaptively: more points in high-curvature regions
    sampled_pts = [control_points[0]]
    i = 0

    while i < len(centerline) - 1:
        curve_ahead = np.mean(np.clip(curvature[i:min(i+20, len(centerline))], 0, 1))
        # Fewer points in curves, more on straights
        step = max(2, int(8 * (1 - 0.5*curve_ahead)))

        if i + step >= len(centerline):
            break
        sampled_pts.append(centerline[i + step])
        i += step

    sampled_pts.append(control_points[-1])
    sampled_pts = np.array(sampled_pts)

    # Add natural jitter (except endpoints)
    jitter = np.random.normal(0, 0.25, sampled_pts.shape)
    jitter[0] = jitter[-1] = 0
    sampled_pts = sampled_pts + jitter

    # Clamp to tunnel boundaries
    for i in range(len(sampled_pts)):
        closest = np.argmin(np.sum((centerline - sampled_pts[i])**2, axis=1))
        cx, cy = centerline[closest]

        # Perpendicular projection
        if closest < len(centerline) - 1:
            perp_x, perp_y = -dy[closest], dx[closest]
            perp_len = np.sqrt(perp_x**2 + perp_y**2) + 1e-8
            perp_x, perp_y = perp_x/perp_len, perp_y/perp_len

            offset = (sampled_pts[i, 0]-cx)*perp_x + (sampled_pts[i, 1]-cy)*perp_y
            offset = np.clip(offset, -half_width+1, half_width-1)

            sampled_pts[i, 0] = cx + offset*perp_x
            sampled_pts[i, 1] = cy + offset*perp_y

    # Realistic timing: variable inter-event intervals
    n_pts = len(sampled_pts)
    total_time = np.random.uniform(1200, 2400)  # 1.2-2.4 seconds

    # Generate inter-event times: mostly 10-18ms, occasional pauses
    dts = np.random.uniform(10, 18, n_pts)
    dts[0] = 0

    # Add ~10% pauses (longer gaps)
    pause_mask = np.random.random(n_pts) < 0.10
    pause_mask[0] = pause_mask[-1] = False
    dts[pause_mask] = np.random.uniform(35, 55, np.sum(pause_mask))

    # Normalize to target total time
    dts_sum = np.sum(dts[1:])
    if dts_sum > 0:
        dts[1:] = dts[1:] / dts_sum * total_time

    timestamps = 210000 + np.cumsum(dts)

    # Build event list
    events = []
    for i, (pt, ts) in enumerate(zip(sampled_pts, timestamps)):
        event_type = "mousedown" if i == 0 else ("mouseup" if i == n_pts-1 else "mousemove")

        # Check tunnel bounds
        min_dist = np.min(np.sqrt(np.sum((centerline - pt)**2, axis=1)))
        inside = min_dist < half_width

        events.append({
            "x": float(pt[0]),
            "y": float(pt[1]),
            "timestamp": float(ts),
            "event_type": event_type,
            "inside_tunnel": inside
        })

    return events
