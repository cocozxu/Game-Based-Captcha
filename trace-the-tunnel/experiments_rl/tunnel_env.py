"""
Python reimplementation of the trace-the-tunnel game engine.

Faithfully ports the JS mulberry32 PRNG and tunnel generation so the
RL environment runs identically to what the browser executes. Validated
by comparing generated control_points against stored human trajectory JSONs.
"""

import ctypes
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants (must match game.js)
# ---------------------------------------------------------------------------
CANVAS_W = 600
CANVAS_H = 350
TUNNEL_WIDTH = 38       # half-width in pixels
DOT_RADIUS = 14
NUM_SEGMENTS = 4
SAMPLES_PER_SEG = 80

# RL configuration
MAX_STEPS = 2000        # max mousemove events per episode
MAX_STEP_PX = 14.0     # max pixel displacement per action step
MIN_STEP_PX = 0.5      # min displacement (prevents zero-move loops)
DT_MEAN_MS = 8.3       # target inter-event delay mean (human)
DT_STD_MS = 0.9        # target inter-event delay std (human)

OBS_DIM = 10


# ---------------------------------------------------------------------------
# Mulberry32 PRNG — exact JS port (32-bit arithmetic)
# ---------------------------------------------------------------------------

def _to_int32(x: int) -> int:
    return ctypes.c_int32(x).value

def _to_uint32(x: int) -> int:
    return x & 0xFFFFFFFF

def _imul32(a: int, b: int) -> int:
    """JS Math.imul: low 32 bits of a×b, returned as signed int32."""
    return ctypes.c_int32((_to_uint32(a) * _to_uint32(b)) & 0xFFFFFFFF).value

def mulberry32(seed: int):
    """Returns a callable RNG matching JS mulberry32(seed)."""
    s = _to_int32(seed)

    def rng() -> float:
        nonlocal s
        s = _to_int32(_to_uint32(s) + 0x6d2b79f5)
        t = _imul32(s ^ (_to_uint32(s) >> 15), 1 | s)
        t = _to_int32(t + _imul32(t ^ (_to_uint32(t) >> 7), 61 | t)) ^ t
        return _to_uint32(t ^ (_to_uint32(t) >> 14)) / 4294967296.0

    return rng


# ---------------------------------------------------------------------------
# Bezier geometry
# ---------------------------------------------------------------------------

@dataclass
class Point:
    x: float
    y: float


@dataclass
class Segment:
    p0: Point
    p1: Point
    p2: Point
    p3: Point


def cubic_bezier(seg: Segment, t: float) -> Point:
    p0, p1, p2, p3 = seg.p0, seg.p1, seg.p2, seg.p3
    u = 1.0 - t
    x = u**3*p0.x + 3*u**2*t*p1.x + 3*u*t**2*p2.x + t**3*p3.x
    y = u**3*p0.y + 3*u**2*t*p1.y + 3*u*t**2*p2.y + t**3*p3.y
    return Point(x, y)


def dist_to_segment_pt(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    len_sq = dx*dx + dy*dy
    if len_sq == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax)*dx + (py - ay)*dy) / len_sq))
    return math.hypot(px - (ax + t*dx), py - (ay + t*dy))


# ---------------------------------------------------------------------------
# Tunnel generation — exact JS port
# ---------------------------------------------------------------------------

def generate_tunnel(seed: int):
    """Returns (segments, control_points, centerline_pts) exactly as game.js."""
    rng = mulberry32(seed)
    margin = 50
    usable_h = CANVAS_H - 2 * margin

    # Knot points
    knots: List[Point] = []
    for i in range(NUM_SEGMENTS + 1):
        frac = i / NUM_SEGMENTS
        knots.append(Point(
            x=margin + frac * (CANVAS_W - 2 * margin),
            y=margin + rng() * usable_h,
        ))

    segments: List[Segment] = []
    prev_handle: Optional[Point] = None

    for i in range(NUM_SEGMENTS):
        a, b = knots[i], knots[i + 1]
        span_y = usable_h * 0.7

        if prev_handle is not None:
            cp1 = Point(x=2*a.x - prev_handle.x, y=2*a.y - prev_handle.y)
        else:
            direction = -1.0 if rng() < 0.5 else 1.0
            cp1 = Point(
                x=a.x + (b.x - a.x) * 0.3,
                y=a.y + direction * (0.3 + rng() * 0.5) * span_y,
            )

        mid_y = (a.y + b.y) / 2
        dir2 = 1.0 if cp1.y < mid_y else -1.0
        cp2 = Point(
            x=a.x + (b.x - a.x) * 0.7,
            y=b.y + dir2 * (0.3 + rng() * 0.5) * span_y,
        )

        low = margin * 0.3
        high = CANVAS_H - margin * 0.3
        cp1.y = max(low, min(high, cp1.y))
        cp2.y = max(low, min(high, cp2.y))

        segments.append(Segment(p0=a, p1=cp1, p2=cp2, p3=b))
        prev_handle = cp2

    # Flat control_points list (matches JS export format)
    control_points = []
    for seg in segments:
        for pt in (seg.p0, seg.p1, seg.p2, seg.p3):
            control_points.append({"x": pt.x, "y": pt.y})

    # Centerline samples
    centerline: List[Point] = []
    for seg in segments:
        for i in range(SAMPLES_PER_SEG + 1):
            if centerline and i == 0:
                continue  # skip duplicate knot
            t = i / SAMPLES_PER_SEG
            centerline.append(cubic_bezier(seg, t))

    return segments, control_points, centerline


# ---------------------------------------------------------------------------
# Tunnel geometry helpers
# ---------------------------------------------------------------------------

def centerline_tangent(px: float, py: float, centerline: List[Point]) -> Tuple[float, float]:
    """Normalized tangent direction of the centerline at the nearest point to (px, py)."""
    min_d = float("inf")
    best_idx = 0
    for i in range(len(centerline) - 1):
        a, b = centerline[i], centerline[i + 1]
        dx, dy = b.x - a.x, b.y - a.y
        len_sq = dx * dx + dy * dy
        if len_sq == 0:
            d = math.hypot(px - a.x, py - a.y)
        else:
            t = max(0.0, min(1.0, ((px - a.x) * dx + (py - a.y) * dy) / len_sq))
            d = math.hypot(px - (a.x + t * dx), py - (a.y + t * dy))
        if d < min_d:
            min_d = d
            best_idx = i
    a, b = centerline[best_idx], centerline[best_idx + 1]
    dx, dy = b.x - a.x, b.y - a.y
    length = math.hypot(dx, dy)
    if length == 0:
        return 1.0, 0.0
    return dx / length, dy / length


def dist_to_centerline(px: float, py: float, centerline: List[Point]) -> float:
    min_d = float("inf")
    for i in range(len(centerline) - 1):
        a, b = centerline[i], centerline[i + 1]
        d = dist_to_segment_pt(px, py, a.x, a.y, b.x, b.y)
        if d < min_d:
            min_d = d
    return min_d


def signed_dist_to_centerline(
    px: float, py: float, centerline: List[Point]
) -> Tuple[float, float]:
    """Returns (signed_distance, arc_progress).

    Sign convention: positive = right of travel direction (looking from start to end).
    arc_progress: fraction [0,1] of total centerline arc.
    """
    min_d = float("inf")
    best_idx = 0
    best_t_local = 0.0

    arcs = [0.0]
    for i in range(len(centerline) - 1):
        a, b = centerline[i], centerline[i + 1]
        arcs.append(arcs[-1] + math.hypot(b.x - a.x, b.y - a.y))

    total_arc = arcs[-1] if arcs[-1] > 0 else 1.0

    for i in range(len(centerline) - 1):
        a, b = centerline[i], centerline[i + 1]
        dx, dy = b.x - a.x, b.y - a.y
        len_sq = dx*dx + dy*dy
        if len_sq == 0:
            d = math.hypot(px - a.x, py - a.y)
            t_local = 0.0
        else:
            t_local = max(0.0, min(1.0, ((px - a.x)*dx + (py - a.y)*dy) / len_sq))
            cx = a.x + t_local*dx
            cy = a.y + t_local*dy
            d = math.hypot(px - cx, py - cy)
        if d < min_d:
            min_d = d
            best_idx = i
            best_t_local = t_local

    # Compute sign using cross product with travel direction at best segment
    a, b = centerline[best_idx], centerline[best_idx + 1]
    seg_dx, seg_dy = b.x - a.x, b.y - a.y
    to_pt_x, to_pt_y = px - (a.x + best_t_local*seg_dx), py - (a.y + best_t_local*seg_dy)
    # cross product z-component (positive = right of direction)
    cross = seg_dx * to_pt_y - seg_dy * to_pt_x
    sign = 1.0 if cross >= 0 else -1.0

    arc_at_best = arcs[best_idx] + best_t_local * (arcs[best_idx + 1] - arcs[best_idx])
    arc_progress = arc_at_best / total_arc

    return sign * min_d, arc_progress


def is_inside_tunnel(px: float, py: float, centerline: List[Point]) -> bool:
    return dist_to_centerline(px, py, centerline) <= TUNNEL_WIDTH


# ---------------------------------------------------------------------------
# RL Environment
# ---------------------------------------------------------------------------

class TunnelEnv:
    """
    Gym-compatible environment for trace-the-tunnel.

    Action space (continuous, 2-dim): (dx, dy) in pixels.
      Scaled by MAX_STEP_PX via tanh output.

    Observation space (10-dim): see _compute_obs().

    Reward:
      - +10.0  task completion (reached red dot)
      - -5.0   leaving tunnel (episode ends)
      - +0.08  per unit of arc-length progress toward goal
      - +0.15  per-step bonus when |signed_cl_dist| ∈ [4, 22] (human-like deviation)
      - -0.01  per step (time penalty)
    """

    metadata = {"render_modes": []}

    def __init__(self, tunnel_id: int, seed: Optional[int] = None):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "tunnel_config.json"
        )
        with open(config_path) as f:
            config = json.load(f)
        entry = next(t for t in config["tunnels"] if t["tunnel_id"] == tunnel_id)
        self.tunnel_seed = entry["seed"]
        self.tunnel_id = tunnel_id

        self._segments, self._control_points, self._centerline = generate_tunnel(
            self.tunnel_seed
        )
        self._start = self._segments[0].p0
        self._end = self._segments[-1].p3

        # arc-length table for the centerline
        self._arcs = [0.0]
        for i in range(len(self._centerline) - 1):
            a, b = self._centerline[i], self._centerline[i + 1]
            self._arcs.append(self._arcs[-1] + math.hypot(b.x - a.x, b.y - a.y))
        self._total_arc = self._arcs[-1]

        self._np_rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self._x = float(self._start.x)
        self._y = float(self._start.y)
        self._vx = 0.0
        self._vy = 0.0
        self._steps = 0
        self._arc_prev = 0.0
        self._done = False
        return self._compute_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        """
        action: (2,) array of (dx, dy) in pixels.
        Returns (obs, reward, terminated, truncated, info).
        """
        dx = float(np.clip(action[0], -MAX_STEP_PX, MAX_STEP_PX))
        dy = float(np.clip(action[1], -MAX_STEP_PX, MAX_STEP_PX))

        # Enforce minimum step size to prevent stalling
        step_len = math.hypot(dx, dy)
        if step_len < MIN_STEP_PX:
            angle = math.atan2(self._end.y - self._y, self._end.x - self._x)
            dx = MIN_STEP_PX * math.cos(angle)
            dy = MIN_STEP_PX * math.sin(angle)

        new_x = self._x + dx
        new_y = self._y + dy

        # Clamp to canvas bounds
        new_x = float(np.clip(new_x, 0, CANVAS_W))
        new_y = float(np.clip(new_y, 0, CANVAS_H))

        self._steps += 1

        # Check tunnel boundary
        if not is_inside_tunnel(new_x, new_y, self._centerline):
            self._done = True
            return self._compute_obs(), -1.0, True, False, {"reason": "out_of_bounds"}

        # Update position
        self._vx = new_x - self._x
        self._vy = new_y - self._y
        self._x, self._y = new_x, new_y

        # Arc-progress reward — primary dense shaping signal
        cl_signed, arc_prog = signed_dist_to_centerline(self._x, self._y, self._centerline)
        arc_now = arc_prog * self._total_arc
        delta_arc = max(0.0, arc_now - self._arc_prev)
        reward = 0.02 * delta_arc     # progress bonus
        reward -= 0.005               # small time penalty to discourage stalling
        self._arc_prev = max(self._arc_prev, arc_now)

        # Soft centerline-deviation shaping: target 5–18 px off-center (human mean ≈ 11px)
        cl_abs = abs(cl_signed)
        if cl_abs > 18.0:
            reward -= 0.02 * (cl_abs - 18.0)   # penalise hugging walls

        # Check goal reached
        goal_dist = math.hypot(self._x - self._end.x, self._y - self._end.y)
        if goal_dist <= DOT_RADIUS + 4:
            self._done = True
            return self._compute_obs(), 1.0 + reward, True, False, {"reason": "success"}

        # Truncation
        if self._steps >= MAX_STEPS:
            return self._compute_obs(), reward, False, True, {"reason": "truncated"}

        return self._compute_obs(), reward, False, False, {}

    # ------------------------------------------------------------------
    def _compute_obs(self) -> np.ndarray:
        """
        10-dimensional observation:
          0: x_norm          = x / CANVAS_W
          1: y_norm          = y / CANVAS_H
          2: vx_norm         = last step dx / MAX_STEP_PX
          3: vy_norm         = last step dy / MAX_STEP_PX
          4: cl_signed_norm  = signed_cl_dist / TUNNEL_WIDTH  (±1 at wall)
          5: goal_dx_norm    = (goal_x - x) / CANVAS_W
          6: goal_dy_norm    = (goal_y - y) / CANVAS_H
          7: goal_dist_norm  = euclidean dist to goal / 700
          8: arc_progress    = fraction of arc completed (0..1)
          9: margin_norm     = (TUNNEL_WIDTH - |cl_dist|) / TUNNEL_WIDTH  (wall clearance)
        """
        cl_signed, arc_prog = signed_dist_to_centerline(self._x, self._y, self._centerline)
        cl_abs = abs(cl_signed)

        gx, gy = self._end.x, self._end.y
        goal_dx = gx - self._x
        goal_dy = gy - self._y
        goal_dist = math.hypot(goal_dx, goal_dy)

        obs = np.array([
            self._x / CANVAS_W,
            self._y / CANVAS_H,
            self._vx / MAX_STEP_PX,
            self._vy / MAX_STEP_PX,
            cl_signed / TUNNEL_WIDTH,
            goal_dx / CANVAS_W,
            goal_dy / CANVAS_H,
            goal_dist / 700.0,
            arc_prog,
            (TUNNEL_WIDTH - cl_abs) / TUNNEL_WIDTH,
        ], dtype=np.float32)

        return np.clip(obs, -3.0, 3.0)

    # ------------------------------------------------------------------
    @property
    def start_pos(self) -> Tuple[float, float]:
        return (self._start.x, self._start.y)

    @property
    def end_pos(self) -> Tuple[float, float]:
        return (self._end.x, self._end.y)

    @property
    def control_points(self):
        return self._control_points

    @property
    def centerline(self) -> List[Point]:
        return self._centerline


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_against_human_data(data_dir: Optional[str] = None):
    """
    Checks that Python-generated control_points match stored human JSONs.
    Prints a per-tunnel max-coordinate error.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "human")

    config_path = os.path.join(os.path.dirname(__file__), "..", "tunnel_config.json")
    with open(config_path) as f:
        config = json.load(f)
    seed_map = {t["tunnel_id"]: t["seed"] for t in config["tunnels"]}

    errors: dict = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(data_dir, fname)) as f:
            traj = json.load(f)
        tid = traj.get("tunnel_id")
        if tid is None or tid not in seed_map:
            continue
        cp_stored = traj.get("control_points", [])
        if not cp_stored:
            continue

        _, cp_gen, _ = generate_tunnel(seed_map[tid])
        if len(cp_gen) != len(cp_stored):
            errors[tid] = f"len mismatch {len(cp_gen)} vs {len(cp_stored)}"
            continue

        max_err = max(
            max(abs(g["x"] - s["x"]), abs(g["y"] - s["y"]))
            for g, s in zip(cp_gen, cp_stored)
        )
        errors.setdefault(tid, max_err)
        errors[tid] = max(errors[tid], max_err)

    all_ok = all(isinstance(v, float) and v < 0.01 for v in errors.values())
    for tid, err in sorted(errors.items()):
        status = "OK" if isinstance(err, float) and err < 0.01 else "FAIL"
        print(f"  tunnel {tid}: max_coord_error={err:.6f}  [{status}]")
    return all_ok


if __name__ == "__main__":
    print("Validating Python tunnel generation against stored human trajectories...")
    ok = validate_against_human_data()
    sys.exit(0 if ok else 1)
