"""
Feature extraction from trace-the-tunnel trajectory JSON files.

Reads trajectory JSONs from data/<source>/ directories and produces a
feature DataFrame suitable for classification.
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def load_trajectories(data_dir):
    """Load all trajectory JSONs from a directory. Returns list of dicts."""
    trajs = []
    if not os.path.isdir(data_dir):
        return trajs
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(data_dir, fname)) as f:
            trajs.append(json.load(f))
    return trajs


def events_to_arrays(events):
    """Extract parallel numpy arrays from the events list."""
    moves = [e for e in events if e["event_type"] == "mousemove"]
    if len(moves) < 2:
        return None
    x = np.array([e["x"] for e in moves])
    y = np.array([e["y"] for e in moves])
    t = np.array([e["timestamp"] for e in moves])
    return x, y, t


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_kinematics(x, y, t):
    """Compute velocity, acceleration, jerk arrays."""
    dt = np.diff(t)
    dt = np.where(dt == 0, 1e-3, dt)  # avoid division by zero

    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx**2 + dy**2) / dt  # pixels per ms

    if len(speed) < 2:
        return speed, np.array([]), np.array([])

    dt2 = dt[:-1]
    dt2 = np.where(dt2 == 0, 1e-3, dt2)
    accel = np.diff(speed) / dt2

    if len(accel) < 2:
        return speed, accel, np.array([])

    dt3 = dt2[:-1]
    dt3 = np.where(dt3 == 0, 1e-3, dt3)
    jerk = np.diff(accel) / dt3

    return speed, accel, jerk


def compute_curvature(x, y):
    """Compute unsigned curvature at each interior point."""
    if len(x) < 3:
        return np.array([])
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5
    denom = np.where(denom == 0, 1e-9, denom)
    kappa = np.abs(dx * ddy - dy * ddx) / denom
    return kappa


def path_length(x, y):
    """Total path length in pixels."""
    return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))


def straight_line_dist(x, y):
    """Euclidean distance from first to last point."""
    return math.hypot(x[-1] - x[0], y[-1] - y[0])


def direction_changes(x, y, threshold_deg=30):
    """Count sharp direction changes above threshold degrees."""
    dx = np.diff(x)
    dy = np.diff(y)
    angles = np.arctan2(dy, dx)
    d_angles = np.abs(np.diff(angles))
    # Wrap to [0, pi]
    d_angles = np.minimum(d_angles, 2 * np.pi - d_angles)
    threshold_rad = np.radians(threshold_deg)
    return int(np.sum(d_angles > threshold_rad))


def compute_centerline_deviation(x, y, control_points):
    """
    Compute mean and std distance from trajectory to tunnel centerline.
    Reconstructs the centerline from control points (groups of 4 = cubic Bezier segments).
    """
    # Reconstruct centerline from control points
    centerline = []
    # control_points is a flat list: [p0,c1,c2,p1, p1,c3,c4,p2, ...]
    num_segs = len(control_points) // 4
    for s in range(num_segs):
        p0 = control_points[s * 4]
        p1 = control_points[s * 4 + 1]
        p2 = control_points[s * 4 + 2]
        p3 = control_points[s * 4 + 3]
        for i in range(50):
            t = i / 49
            u = 1 - t
            cx = u**3*p0["x"] + 3*u**2*t*p1["x"] + 3*u*t**2*p2["x"] + t**3*p3["x"]
            cy = u**3*p0["y"] + 3*u**2*t*p1["y"] + 3*u*t**2*p2["y"] + t**3*p3["y"]
            centerline.append((cx, cy))

    centerline = np.array(centerline)

    # For each trajectory point, find min distance to centerline
    dists = []
    for xi, yi in zip(x, y):
        d = np.sqrt((centerline[:, 0] - xi)**2 + (centerline[:, 1] - yi)**2)
        dists.append(d.min())

    dists = np.array(dists)
    return dists.mean(), dists.std()


def safe_stats(arr):
    """Return mean, std, min, max, median for an array, or zeros if empty."""
    if len(arr) == 0:
        return 0, 0, 0, 0, 0
    return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr)), float(np.median(arr))


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

def extract_features(traj):
    """
    Extract a feature dict from a single trajectory JSON.
    Returns None if the trajectory has insufficient data.
    """
    events = traj.get("events", [])
    result = events_to_arrays(events)
    if result is None:
        return None

    x, y, t = result
    n_points = len(x)
    if n_points < 5:
        return None

    # Normalize time to start at 0
    t = t - t[0]

    # --- Kinematic features ---
    speed, accel, jerk = compute_kinematics(x, y, t)
    spd_mean, spd_std, spd_min, spd_max, spd_med = safe_stats(speed)
    acc_mean, acc_std, acc_min, acc_max, acc_med = safe_stats(accel)
    jrk_mean, jrk_std, jrk_min, jrk_max, jrk_med = safe_stats(jerk)

    # --- Curvature features ---
    kappa = compute_curvature(x, y)
    curv_mean, curv_std, curv_min, curv_max, curv_med = safe_stats(kappa)

    # --- Temporal features ---
    dt = np.diff(t)
    dt_mean, dt_std, dt_min, dt_max, dt_med = safe_stats(dt)

    # --- Geometric features ---
    plen = path_length(x, y)
    sld = straight_line_dist(x, y)
    efficiency = sld / plen if plen > 0 else 0
    n_dir_changes = direction_changes(x, y)

    # --- Centerline deviation ---
    cp = traj.get("control_points", [])
    if len(cp) >= 4:
        cl_dev_mean, cl_dev_std = compute_centerline_deviation(x, y, cp)
    else:
        cl_dev_mean, cl_dev_std = 0, 0

    # --- Duration ---
    duration_ms = t[-1] - t[0]

    # --- Frequency domain: tremor band power ---
    # Resample speed to uniform spacing for FFT
    tremor_power = 0.0
    if len(speed) > 20 and dt_mean > 0:
        try:
            fs = 1000.0 / dt_mean  # approx sampling rate in Hz
            freqs = np.fft.rfftfreq(len(speed), d=1.0/fs)
            fft_mag = np.abs(np.fft.rfft(speed - speed.mean()))
            # Tremor band: 8-12 Hz (physiological hand tremor)
            mask = (freqs >= 8) & (freqs <= 12)
            if mask.any():
                tremor_power = float(np.mean(fft_mag[mask]**2))
        except Exception:
            pass

    return {
        # Metadata
        "session_id": traj.get("session_id"),
        "tunnel_id": traj.get("tunnel_id"),
        "source": traj.get("source", "unknown"),
        "completed": traj.get("completed", False),
        "fail_reason": traj.get("fail_reason"),
        "n_points": n_points,
        "duration_ms": duration_ms,

        # Speed
        "speed_mean": spd_mean,
        "speed_std": spd_std,
        "speed_min": spd_min,
        "speed_max": spd_max,
        "speed_median": spd_med,

        # Acceleration
        "accel_mean": acc_mean,
        "accel_std": acc_std,
        "accel_max": acc_max,

        # Jerk
        "jerk_mean": jrk_mean,
        "jerk_std": jrk_std,
        "jerk_max": jrk_max,

        # Curvature
        "curvature_mean": curv_mean,
        "curvature_std": curv_std,
        "curvature_max": curv_max,

        # Timing
        "dt_mean": dt_mean,
        "dt_std": dt_std,
        "dt_min": dt_min,
        "dt_max": dt_max,

        # Geometry
        "path_length": plen,
        "straight_line_dist": sld,
        "path_efficiency": efficiency,
        "direction_changes": n_dir_changes,

        # Centerline deviation
        "centerline_dev_mean": cl_dev_mean,
        "centerline_dev_std": cl_dev_std,

        # Frequency
        "tremor_power_8_12hz": tremor_power,
    }


def build_feature_dataframe(base_data_dir):
    """
    Scan all source directories under base_data_dir, extract features,
    return a pandas DataFrame.
    """
    rows = []
    for source_dir in sorted(os.listdir(base_data_dir)):
        full_path = os.path.join(base_data_dir, source_dir)
        if not os.path.isdir(full_path) or source_dir == "logs":
            continue
        trajs = load_trajectories(full_path)
        for traj in trajs:
            feat = extract_features(traj)
            if feat is not None:
                rows.append(feat)

    df = pd.DataFrame(rows)
    # Add binary label: 0 = human, 1 = agent (any kind)
    if "source" in df.columns:
        df["is_agent"] = (~df["source"].fillna("human").str.startswith("human")).astype(int)
    return df


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    df = build_feature_dataframe(base_dir)
    print(f"Extracted {len(df)} trajectories")
    print(f"Sources: {df['source'].value_counts().to_dict()}")
    print(f"Completed: {df['completed'].value_counts().to_dict()}")
    out_path = os.path.join(os.path.dirname(__file__), "features.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
