def generate(tunnel_spec: dict, seed: int) -> list[dict]:
    """
    Generate human-like mouse traces for tunnel navigation by:
    1. Smoothly interpolating the tunnel centerline via Catmull-Rom spline
    2. Computing curvature to modulate jitter magnitude
    3. Adding realistic micro-corrections perpendicular to the path
    4. Generating natural timing with speed-dependent dt distribution
    """
    import numpy as np

    np.random.seed(seed)

    control_points = np.array([[p['x'], p['y']] for p in tunnel_spec['control_points']])
    tunnel_width = tunnel_spec['tunnel_width']
    half_width = tunnel_width / 2.0

    # Remove consecutive duplicates from control points
    unique = [control_points[0]]
    for p in control_points[1:]:
        if not np.allclose(p, unique[-1], atol=1e-6):
            unique.append(p)
    control_points = np.array(unique)

    # Interpolate smooth centerline using Catmull-Rom spline
    centerline = catmull_rom_interpolate(control_points, num_points=280)

    # Estimate curvature for adaptive jitter
    curvature = estimate_curvature(centerline)

    # Add human-like jitter with curvature-dependent magnitude
    trajectory = centerline.copy().astype(np.float64)
    for i in range(1, len(centerline) - 1):
        direction = centerline[i+1] - centerline[i-1]
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-6:
            direction = direction / direction_norm
            perpendicular = np.array([-direction[1], direction[0]])

            # Jitter increases with curvature (more corrections on sharp turns)
            jitter_magnitude = 0.3 + 1.5 * curvature[i]
            jitter_amount = np.random.normal(0, jitter_magnitude)
            trajectory[i] = trajectory[i] + perpendicular * jitter_amount

    # Generate mouse events with natural timing
    events = []
    timestamp = 0.0

    # Mousedown at start
    events.append({
        'x': float(trajectory[0][0]),
        'y': float(trajectory[0][1]),
        'timestamp': timestamp,
        'event_type': 'mousedown',
        'inside_tunnel': True
    })

    # Mousemove events with speed-dependent dt
    for i in range(1, len(trajectory)):
        distance = np.linalg.norm(trajectory[i] - trajectory[i-1])

        # dt inversely related to distance (faster movement = shorter dt)
        if distance < 0.05:
            dt = np.random.exponential(4.0) + 6.0
        elif distance < 0.5:
            dt = np.random.normal(8.5, 2.5)
        else:
            dt = np.random.normal(7.5, 3.0)

        # Clamp dt to reasonable range
        dt = max(4.0, min(25.0, dt))
        timestamp += dt

        events.append({
            'x': float(trajectory[i][0]),
            'y': float(trajectory[i][1]),
            'timestamp': timestamp,
            'event_type': 'mousemove',
            'inside_tunnel': True
        })

    # Mouseup at end
    events[-1]['event_type'] = 'mouseup'

    return events


def catmull_rom_interpolate(points, num_points=280):
    """Interpolate control points using Catmull-Rom spline basis functions."""
    import numpy as np

    if len(points) < 2:
        return points

    interpolated = []
    num_segments = len(points) - 1
    points_per_segment = max(1, num_points // num_segments)

    for i in range(len(points) - 1):
        # Get neighboring control points for this segment
        p0 = points[max(0, i - 1)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(len(points) - 1, i + 2)]

        # Generate interpolated points along this segment
        for j in range(points_per_segment):
            t = j / max(1, points_per_segment)
            t2 = t * t
            t3 = t2 * t

            # Catmull-Rom basis matrix coefficients
            basis = np.array([
                -0.5 * t3 + t2 - 0.5 * t,
                1.5 * t3 - 2.5 * t2 + 1.0,
                -1.5 * t3 + 2.0 * t2 + 0.5 * t,
                0.5 * t3 - 0.5 * t2
            ])

            point = basis[0] * p0 + basis[1] * p1 + basis[2] * p2 + basis[3] * p3
            interpolated.append(point)

    # Add the final point
    interpolated.append(points[-1].astype(np.float64))

    return np.array(interpolated)


def estimate_curvature(path):
    """Estimate local curvature at each point along the path."""
    import numpy as np

    n = len(path)
    curvature = np.zeros(n)

    for i in range(1, n - 1):
        # Tangent vectors
        v1 = path[i] - path[i - 1]
        v2 = path[i + 1] - path[i]

        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        # Curvature via angle between tangents divided by arc length
        if len1 > 1e-6 and len2 > 1e-6:
            cos_angle = np.dot(v1, v2) / (len1 * len2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            curvature[i] = angle / (len1 + len2)

    # Smooth curvature with moving average
    for _ in range(2):
        smoothed = np.copy(curvature)
        for i in range(1, n - 1):
            smoothed[i] = (curvature[i - 1] + 2.0 * curvature[i] + curvature[i + 1]) / 4.0
        curvature = smoothed

    return curvature
