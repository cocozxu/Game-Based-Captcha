"""Debug harness for the warp solver.

Mirrors the algorithm in warp_solver.js end-to-end so we can characterize
the artifacts that the in-browser smoketest doesn't catch: non-monotonic
arc-length sequences, segment flips at sharp bends, out-of-tunnel events,
and how the tail-snap fires. It can run both the CURRENT (buggy) algorithm
and the PROPOSED FIX so we can quote before/after numbers.

Usage:
  python experiments_warp/_debug_warp.py
"""

import json
import math
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))


def cubic_bezier(p0, p1, p2, p3, t):
    u = 1 - t
    return (
        u*u*u*p0["x"] + 3*u*u*t*p1["x"] + 3*u*t*t*p2["x"] + t*t*t*p3["x"],
        u*u*u*p0["y"] + 3*u*u*t*p1["y"] + 3*u*t*t*p2["y"] + t*t*t*p3["y"],
    )


def build_centerline(control_points, samples_per_seg=50):
    pts = []
    num_segs = len(control_points) // 4
    for s in range(num_segs):
        p0, p1, p2, p3 = control_points[s*4: s*4+4]
        for i in range(samples_per_seg):
            t = i / (samples_per_seg - 1)
            pts.append(cubic_bezier(p0, p1, p2, p3, t))
    cum = [0.0]
    for i in range(1, len(pts)):
        ax, ay = pts[i-1]; bx, by = pts[i]
        cum.append(cum[-1] + math.hypot(bx-ax, by-ay))
    total = cum[-1] or 1.0
    frac = [c/total for c in cum]
    return pts, cum, frac, total


def project_global(px, py, cl):
    """Closest-point projection on the full polyline (current behavior)."""
    pts, cum, frac, total = cl
    best_d2 = float("inf"); best_idx = 0; best_t = 0.0
    best_cx = pts[0][0]; best_cy = pts[0][1]
    for i in range(len(pts) - 1):
        ax, ay = pts[i]; bx, by = pts[i+1]
        dx, dy = bx-ax, by-ay
        len_sq = dx*dx + dy*dy
        if len_sq == 0:
            t = 0.0
        else:
            t = ((px-ax)*dx + (py-ay)*dy) / len_sq
            t = max(0.0, min(1.0, t))
        cx = ax + t*dx; cy = ay + t*dy
        d2 = (px-cx)**2 + (py-cy)**2
        if d2 < best_d2:
            best_d2 = d2; best_idx = i; best_t = t; best_cx = cx; best_cy = cy
    arc = cum[best_idx] + best_t * (cum[best_idx+1] - cum[best_idx])
    f = arc / total
    ax, ay = pts[best_idx]; bx, by = pts[best_idx+1]
    tdx, tdy = bx-ax, by-ay
    tn = math.hypot(tdx, tdy) or 1
    tdx /= tn; tdy /= tn
    vx, vy = px - best_cx, py - best_cy
    cross = tdx*vy - tdy*vx
    dist = math.sqrt(best_d2)
    d = dist if cross >= 0 else -dist
    return f, d, best_idx


def project_local(px, py, cl, prev_idx, window=12):
    """Windowed closest-point projection (proposed fix for H2).

    Restrict the segment search to [prev_idx - 4, prev_idx + 8],
    which keeps the projection locally monotonic across sharp source bends.
    The window must be tight enough to bracket only the local centerline
    branch — a wider window re-admits the U-bend branch flip.
    """
    pts, cum, frac, total = cl
    n = len(pts) - 1
    if prev_idx < 0:
        lo = 0; hi = n
    else:
        lo = max(0, prev_idx - 4)
        hi = min(n, prev_idx + 8)
    best_d2 = float("inf"); best_idx = lo; best_t = 0.0
    best_cx = pts[lo][0]; best_cy = pts[lo][1]
    for i in range(lo, hi):
        ax, ay = pts[i]; bx, by = pts[i+1]
        dx, dy = bx-ax, by-ay
        len_sq = dx*dx + dy*dy
        if len_sq == 0:
            t = 0.0
        else:
            t = ((px-ax)*dx + (py-ay)*dy) / len_sq
            t = max(0.0, min(1.0, t))
        cx = ax + t*dx; cy = ay + t*dy
        d2 = (px-cx)**2 + (py-cy)**2
        if d2 < best_d2:
            best_d2 = d2; best_idx = i; best_t = t; best_cx = cx; best_cy = cy
    arc = cum[best_idx] + best_t * (cum[best_idx+1] - cum[best_idx])
    f = arc / total
    ax, ay = pts[best_idx]; bx, by = pts[best_idx+1]
    tdx, tdy = bx-ax, by-ay
    tn = math.hypot(tdx, tdy) or 1
    tdx /= tn; tdy /= tn
    vx, vy = px - best_cx, py - best_cy
    cross = tdx*vy - tdy*vx
    dist = math.sqrt(best_d2)
    d = dist if cross >= 0 else -dist
    return f, d, best_idx


def sample_at_fraction(f, cl):
    pts, cum, frac, total = cl
    L = len(pts)
    if f <= 0:
        tdx = pts[1][0] - pts[0][0]; tdy = pts[1][1] - pts[0][1]
        n = math.hypot(tdx, tdy) or 1
        return pts[0][0], pts[0][1], tdx/n, tdy/n
    if f >= 1:
        tdx = pts[L-1][0] - pts[L-2][0]; tdy = pts[L-1][1] - pts[L-2][1]
        n = math.hypot(tdx, tdy) or 1
        return pts[L-1][0], pts[L-1][1], tdx/n, tdy/n
    lo, hi = 0, L - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if frac[mid] <= f:
            lo = mid
        else:
            hi = mid
    denom = (frac[hi] - frac[lo]) or 1e-9
    u = (f - frac[lo]) / denom
    x = pts[lo][0] + u * (pts[hi][0] - pts[lo][0])
    y = pts[lo][1] + u * (pts[hi][1] - pts[lo][1])
    tdx = pts[hi][0] - pts[lo][0]; tdy = pts[hi][1] - pts[lo][1]
    n = math.hypot(tdx, tdy) or 1
    return x, y, tdx/n, tdy/n


def dist_to_centerline(px, py, cl):
    pts = cl[0]
    best = float("inf")
    for i in range(len(pts) - 1):
        ax, ay = pts[i]; bx, by = pts[i+1]
        dx, dy = bx-ax, by-ay
        len_sq = dx*dx + dy*dy
        if len_sq == 0:
            d = math.hypot(px-ax, py-ay)
        else:
            t = max(0, min(1, ((px-ax)*dx + (py-ay)*dy) / len_sq))
            cx = ax + t*dx; cy = ay + t*dy
            d = math.hypot(px-cx, py-cy)
        if d < best: best = d
    return best


def warp(events, src_cp, live_cp, tunnel_width, mode):
    """Run the warp algorithm in `mode` and return per-event diagnostic rows.

    mode:
      "current"  -> emulate the BUGGY warp_solver.js (global projection,
                    closest-point f, hit-radius snap)
      "fixed"    -> emulate the FIXED warp_solver.js: cumulative-event-path f
                    (H1), windowed local projection for lateral d only (H2),
                    arc-fraction tail-snap at f>=0.97 (H4).
    """
    cl_src = build_centerline(src_cp)
    cl_live = build_centerline(live_cp)
    live_red = live_cp[-1]
    src_red = src_cp[-1]
    HIT_RADIUS = 18

    snap_from = -1
    if mode == "current":
        for i, ev in enumerate(events):
            if math.hypot(ev["x"] - src_red["x"], ev["y"] - src_red["y"]) <= HIT_RADIUS:
                snap_from = i
                break

    # Precompute cumulative event-path length for fixed mode.
    cum_ev = [0.0]
    for i in range(1, len(events)):
        dx = events[i]["x"] - events[i-1]["x"]
        dy = events[i]["y"] - events[i-1]["y"]
        cum_ev.append(cum_ev[-1] + math.hypot(dx, dy))
    total_ev = cum_ev[-1] or 1.0

    rows = []
    idx_prev = -1
    for i, ev in enumerate(events):
        if mode == "current":
            if snap_from >= 0 and i >= snap_from:
                rows.append({"i": i, "f": 1.0, "d": 0.0, "idx": -1,
                             "wx": live_red["x"], "wy": live_red["y"], "snap": True})
                continue
            f, d, idx = project_global(ev["x"], ev["y"], cl_src)
        else:
            f = cum_ev[i] / total_ev
            if f >= 0.97:
                rows.append({"i": i, "f": 1.0, "d": 0.0, "idx": idx_prev,
                             "wx": live_red["x"], "wy": live_red["y"], "snap": True})
                continue
            # Option D: lateral d = perpendicular distance from event to
            # source-centerline-at-f (not closest projection). idx tracking
            # is unused — kept for the row schema.
            sx, sy, stx, sty = sample_at_fraction(f, cl_src)
            snx, sny = -sty, stx
            d = (ev["x"] - sx) * snx + (ev["y"] - sy) * sny
            idx = 0
            idx_prev = idx

        x, y, tx, ty = sample_at_fraction(f, cl_live)
        nx, ny = -ty, tx
        wx = x + d * nx
        wy = y + d * ny
        # Match the JS clip behavior so we count out-of-tunnel consistently.
        if abs(d) > tunnel_width:
            margin = 2
            shrunk = math.copysign(max(0, tunnel_width - margin), d)
            wx = x + shrunk * nx
            wy = y + shrunk * ny
        rows.append({"i": i, "f": f, "d": d, "idx": idx, "wx": wx, "wy": wy, "snap": False})
    return rows, cl_live


def analyze(rows, cl_live, tunnel_width, label):
    fs = [r["f"] for r in rows]
    idxs = [r["idx"] for r in rows]
    # Non-monotonic jumps (decreases) — excluding snap rows
    non_mono = 0
    big_back = 0
    for i in range(1, len(rows)):
        if rows[i]["snap"]:
            continue
        if fs[i] < fs[i-1] - 1e-9:
            non_mono += 1
            if (fs[i-1] - fs[i]) > 0.02:
                big_back += 1
    # Segment flips
    seg_flips = 0
    for i in range(1, len(rows)):
        if rows[i]["snap"] or rows[i-1]["snap"]:
            continue
        if abs(idxs[i] - idxs[i-1]) > 2:
            seg_flips += 1
    # Out of tunnel events (warped point)
    out = 0
    for r in rows:
        d = dist_to_centerline(r["wx"], r["wy"], cl_live)
        if d > tunnel_width:
            out += 1
    # Direction-changes — matches analysis/features.py:direction_changes (30deg threshold)
    dir_flips = 0
    angles = []
    for i in range(1, len(rows)):
        dx = rows[i]["wx"] - rows[i-1]["wx"]
        dy = rows[i]["wy"] - rows[i-1]["wy"]
        angles.append(math.atan2(dy, dx))
    thr = math.radians(30)
    for i in range(1, len(angles)):
        da = abs(angles[i] - angles[i-1])
        da = min(da, 2*math.pi - da)
        if da > thr:
            dir_flips += 1
    # Acute reversals (>120 deg) — the visible zigzag artifact
    big_turns = 0
    max_ang = 0.0
    for i in range(2, len(rows)):
        ax = rows[i-2]["wx"]; ay = rows[i-2]["wy"]
        bx = rows[i-1]["wx"]; by_ = rows[i-1]["wy"]
        cx = rows[i]["wx"]; cy = rows[i]["wy"]
        v1x = bx-ax; v1y = by_-ay
        v2x = cx-bx; v2y = cy-by_
        l1 = math.hypot(v1x, v1y); l2 = math.hypot(v2x, v2y)
        if l1 < 0.5 or l2 < 0.5: continue
        co = (v1x*v2x + v1y*v2y)/(l1*l2)
        co = max(-1, min(1, co))
        ang = math.degrees(math.acos(co))
        if ang > max_ang: max_ang = ang
        if ang > 120: big_turns += 1
    # Speed std on warped (no real time; just step length over event index)
    steps = []
    for i in range(1, len(rows)):
        dx = rows[i]["wx"]-rows[i-1]["wx"]
        dy = rows[i]["wy"]-rows[i-1]["wy"]
        steps.append(math.hypot(dx, dy))
    mean = sum(steps)/len(steps) if steps else 0
    var = sum((s-mean)**2 for s in steps)/len(steps) if steps else 0
    speed_std = math.sqrt(var)
    print(f"  [{label:>8s}]  n={len(rows):>3d}  "
          f"non_mono_f={non_mono:>3d}  big_back(>2%)={big_back:>3d}  "
          f"seg_flips={seg_flips:>3d}  oot={out:>3d}  "
          f"dir_chg(30)={dir_flips:>3d}  "
          f"big_turn(>120)={big_turns:>2d}  max_ang={max_ang:5.1f}  "
          f"step_std={speed_std:.2f}")
    return {"non_mono": non_mono, "big_back": big_back, "seg_flips": seg_flips,
            "out_of_tunnel": out, "dir_flips": dir_flips, "step_std": speed_std,
            "big_turns": big_turns, "max_ang": max_ang}


def pick_pair():
    allow_path = os.path.join(REPO, "experiments_replay", "allowed_sessions_v1b.json")
    with open(allow_path) as f:
        allow = set(json.load(f))
    human_dir = os.path.join(REPO, "data", "human")
    by_tid = {}
    for sid in allow:
        fp = os.path.join(human_dir, sid + ".json")
        if not os.path.isfile(fp):
            continue
        with open(fp) as f:
            d = json.load(f)
        if d.get("completed"):
            by_tid.setdefault(d["tunnel_id"], []).append(d)
    tids = sorted(by_tid.keys())
    return by_tid, tids


def main():
    by_tid, tids = pick_pair()
    print(f"Discovered tunnels: {tids}")

    # Pair selection. The first pair is the KNOWN BAD case: src=tid 4 source
    # 8d3a0046 -> live tid 6 (matches the 4ce03d56 warp_v1 output with a
    # 180-deg branch flip at i=53). The rest sample other (source, live) pairs.
    fname_src = "8d3a0046-3215-49b1-a528-dc493e5b4303.json"
    fpath = os.path.join(REPO, "data", "human", fname_src)
    known_bad_src = None
    if os.path.isfile(fpath):
        known_bad_src = json.load(open(fpath))
    pairs = []
    if known_bad_src is not None:
        # Find any live tunnel 6 trace from our by_tid dict
        if 6 in by_tid:
            pairs.append(("known_bad", known_bad_src, by_tid[6][0]))
    pairs += [
        ("p1", by_tid[tids[0]][0], by_tid[tids[1]][0]),
        ("p2", by_tid[tids[1]][0], by_tid[tids[2]][0]),
        ("p3", by_tid[tids[3]][0], by_tid[tids[5] if len(tids) > 5 else tids[2]][0]),
        ("p4", by_tid[tids[2]][0], by_tid[tids[7] if len(tids) > 7 else tids[1]][0]),
    ]
    width = 38
    keys = ("non_mono","big_back","seg_flips","out_of_tunnel","dir_flips","big_turns","max_ang","step_std")
    agg_cur = {k:0.0 for k in keys}; agg_cur["n"] = 0
    agg_fix = {k:0.0 for k in keys}; agg_fix["n"] = 0
    for (label, src, live) in pairs:
        # Skip self-pair degeneracy
        if src["tunnel_id"] == live["tunnel_id"]:
            continue
        print(f"\n--- [{label}] src tid={src['tunnel_id']} sid={src['session_id'][:8]} (n={len(src['events'])})  "
              f"-> live tid={live['tunnel_id']} sid={live['session_id'][:8]} ---")
        rows_cur, cl_live = warp(src["events"], src["control_points"], live["control_points"], width, "current")
        rows_fix, _ = warp(src["events"], src["control_points"], live["control_points"], width, "fixed")
        a_cur = analyze(rows_cur, cl_live, width, "current")
        a_fix = analyze(rows_fix, cl_live, width, "fixed")
        for k in keys:
            agg_cur[k] += a_cur[k]
            agg_fix[k] += a_fix[k]
        agg_cur["n"] += 1
        agg_fix["n"] += 1
        tail_fs = [r["f"] for r in rows_cur[-8:]]
        print(f"      tail f (current, last 8): {[f'{x:.3f}' for x in tail_fs]}")

    print("\n=== TOTAL (sums across pairs) ===")
    for k in ("non_mono","big_back","seg_flips","out_of_tunnel","dir_flips","big_turns"):
        print(f"  {k:>15s}:  current={int(agg_cur[k]):>4d}   fixed={int(agg_fix[k]):>4d}")
    n_cur = agg_cur["n"] or 1; n_fix = agg_fix["n"] or 1
    print(f"  {'max_ang_avg':>15s}:  current={agg_cur['max_ang']/n_cur:5.1f}   fixed={agg_fix['max_ang']/n_fix:5.1f}")
    print(f"  {'step_std_mean':>15s}:  current={agg_cur['step_std']/n_cur:.2f}   fixed={agg_fix['step_std']/n_fix:.2f}")


if __name__ == "__main__":
    main()
