"""
Human-like cursor path generator.

Takes a sequence of centerline waypoints and produces a new path that
mimics human motor behavior:
  - Bezier-smoothed sub-paths with slight overshoot
  - Speed profile following a minimum-jerk model (slow-fast-slow)
  - Gaussian noise / micro-corrections perpendicular to the path
  - Variable inter-move delays

Usage as CLI (for Claude Code to call via Bash):
    python human_cursor.py --centerline '[[x,y], ...]' [--noise 2.0] [--overshoots 3]

Outputs JSON list of {x, y, delay_ms} waypoints to stdout.
"""

import argparse
import json
import math
import random
import sys

import numpy as np
from scipy.interpolate import splprep, splev


def perpendicular_offset(p0, p1, magnitude):
    """Return a point offset perpendicular to the segment p0->p1."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (0.0, 0.0)
    # Perpendicular unit vector
    nx = -dy / length
    ny = dx / length
    return (nx * magnitude, ny * magnitude)


def add_noise(points, noise_std=2.0):
    """Add Gaussian noise perpendicular to the path direction."""
    noisy = [points[0]]  # keep start exact
    for i in range(1, len(points) - 1):
        prev = points[i - 1]
        curr = points[i]
        nxt = points[i + 1]
        # Use average direction of neighboring segments
        dx = nxt[0] - prev[0]
        dy = nxt[1] - prev[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            noisy.append(curr)
            continue
        nx = -dy / length
        ny = dx / length
        offset = random.gauss(0, noise_std)
        noisy.append((curr[0] + nx * offset, curr[1] + ny * offset))
    noisy.append(points[-1])  # keep end exact
    return noisy


def add_overshoots(points, num_overshoots=2, magnitude_range=(3.0, 8.0)):
    """Insert small overshoots at random positions along the path."""
    if len(points) < 10 or num_overshoots <= 0:
        return points

    result = list(points)
    indices = sorted(random.sample(range(len(points) // 4, 3 * len(points) // 4),
                                    min(num_overshoots, len(points) // 4)),
                     reverse=True)

    for idx in indices:
        if idx <= 0 or idx >= len(result) - 1:
            continue
        prev = result[idx - 1]
        curr = result[idx]
        mag = random.uniform(*magnitude_range)
        sign = random.choice([-1, 1])
        px, py = perpendicular_offset(prev, curr, mag * sign)
        # Insert overshoot point then correction back
        overshoot = (curr[0] + px, curr[1] + py)
        result.insert(idx + 1, overshoot)

    return result


def minimum_jerk_timing(n_points, total_time_ms=800):
    """
    Generate delays between points following a minimum-jerk speed profile.
    Speed is slow at start/end, fast in the middle — matches human reaching.
    """
    # Minimum jerk speed profile: v(t) = 30*t^2 - 60*t^3 + 30*t^4  (normalized)
    t_norm = np.linspace(0, 1, n_points)
    speed = 30 * t_norm**2 - 60 * t_norm**3 + 30 * t_norm**4
    speed = np.clip(speed, 0.05, None)  # avoid division by zero

    # Time between points is inversely proportional to speed
    inv_speed = 1.0 / speed
    inv_speed_sum = inv_speed.sum()
    delays = (inv_speed / inv_speed_sum) * total_time_ms

    # Add small jitter
    jitter = np.random.normal(0, 1.5, n_points)
    delays = np.clip(delays + jitter, 2, None)

    return delays.tolist()


def smooth_path(points, num_output=150):
    """Resample path using B-spline for smooth curvature."""
    pts = np.array(points)
    # Need at least 4 points for cubic spline
    if len(pts) < 4:
        return points

    try:
        tck, u = splprep([pts[:, 0], pts[:, 1]], s=len(pts) * 2, k=3)
        u_new = np.linspace(0, 1, num_output)
        x_new, y_new = splev(u_new, tck)
        return list(zip(x_new.tolist(), y_new.tolist()))
    except Exception:
        return points


def generate_human_path(centerline, noise_std=2.0, num_overshoots=2,
                         total_time_ms=800, num_output=150):
    """
    Full pipeline: centerline -> human-like path with timing.

    Returns list of {x, y, delay_ms} dicts.
    """
    points = [(p[0], p[1]) for p in centerline]

    # 1. Add overshoots at random points
    points = add_overshoots(points, num_overshoots=num_overshoots)

    # 2. Smooth with B-spline
    points = smooth_path(points, num_output=num_output)

    # 3. Add perpendicular noise (micro-corrections)
    points = add_noise(points, noise_std=noise_std)

    # 4. Generate minimum-jerk timing
    delays = minimum_jerk_timing(len(points), total_time_ms=total_time_ms)

    return [{"x": round(p[0], 2), "y": round(p[1], 2), "delay_ms": round(d, 1)}
            for p, d in zip(points, delays)]


def main():
    parser = argparse.ArgumentParser(description="Generate human-like cursor path")
    parser.add_argument("--centerline", required=True,
                        help="JSON array of [x,y] points")
    parser.add_argument("--noise", type=float, default=2.0,
                        help="Std dev of perpendicular noise (pixels)")
    parser.add_argument("--overshoots", type=int, default=2,
                        help="Number of overshoot points to insert")
    parser.add_argument("--duration", type=float, default=800,
                        help="Target total trace duration in ms")
    parser.add_argument("--points", type=int, default=150,
                        help="Number of output waypoints")

    args = parser.parse_args()
    centerline = json.loads(args.centerline)
    path = generate_human_path(
        centerline,
        noise_std=args.noise,
        num_overshoots=args.overshoots,
        total_time_ms=args.duration,
        num_output=args.points,
    )
    json.dump(path, sys.stdout)


if __name__ == "__main__":
    main()
