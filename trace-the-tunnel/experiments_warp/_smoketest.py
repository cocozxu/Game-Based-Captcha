"""Smoke tests for the warp experiment scaffolding.

Test 1: server endpoint — confirm /api/human_trace_full/<sid> returns
control_points + events for an allowlisted session and 404s for an
unknown one. Uses Flask test_client (no real server bind needed).

Test 2: warp math — reconstruct centerlines from two different humans'
control_points, warp source events onto the live tunnel, and check
(a) endpoints land near the live tunnel anchors, (b) coordinates stay
bounded, (c) >=95% of warped points fall within tunnel_width of the
live centerline.

Run from the repo root:
  source .venv/bin/activate
  python experiments_warp/_smoketest.py
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)

import server  # noqa: E402


# --- Test 1: endpoint --------------------------------------------------------

def test_endpoint():
    print("=== Test 1: /api/human_trace_full/<sid> ===")
    allowlist_path = os.path.join(REPO, "experiments_replay", "allowed_sessions_v1b.json")
    with open(allowlist_path) as f:
        allowlist = json.load(f)
    server.ALLOWED_SESSIONS = set(allowlist)

    client = server.app.test_client()

    # Pick a known allowlist session — must be a completed human file in data/human/.
    human_dir = os.path.join(REPO, "data", "human")
    known_sid = None
    known_tid = None
    for sid in allowlist:
        fpath = os.path.join(human_dir, f"{sid}.json")
        if os.path.isfile(fpath):
            with open(fpath) as f:
                d = json.load(f)
            if d.get("completed"):
                known_sid = sid
                known_tid = d.get("tunnel_id")
                break
    assert known_sid, "no allowlist session found on disk"

    r = client.get(f"/api/human_trace_full/{known_sid}")
    print(f"  GET /api/human_trace_full/{known_sid[:8]}... -> {r.status_code}")
    assert r.status_code == 200
    data = r.get_json()
    assert data["session_id"] == known_sid
    assert data["tunnel_id"] == known_tid
    assert len(data["control_points"]) >= 4
    assert len(data["events"]) >= 5
    print(f"    tunnel_id={data['tunnel_id']}  control_points={len(data['control_points'])}  events={len(data['events'])}")

    # Unknown sid (random UUID-shape) must 404
    r2 = client.get("/api/human_trace_full/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    print(f"  GET /api/human_trace_full/aaaaaaaa-... -> {r2.status_code}")
    assert r2.status_code == 404

    # Session not in allowlist should 404 (try a non-allowlisted completed human)
    other_sid = None
    for fname in sorted(os.listdir(human_dir)):
        sid = fname.replace(".json", "")
        if sid not in server.ALLOWED_SESSIONS:
            with open(os.path.join(human_dir, fname)) as f:
                if json.load(f).get("completed"):
                    other_sid = sid
                    break
    if other_sid:
        r3 = client.get(f"/api/human_trace_full/{other_sid}")
        print(f"  GET /api/human_trace_full/{other_sid[:8]}... (not in allowlist) -> {r3.status_code}")
        assert r3.status_code == 404
    else:
        print("  (no non-allowlist completed human on disk to test allowlist 404; skipped)")

    # Reset module state
    server.ALLOWED_SESSIONS = None
    print("  Test 1 PASS")
    return known_sid, known_tid


# --- Test 2: warp math -------------------------------------------------------

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


def project_to_centerline(px, py, cl):
    pts, cum, frac, total = cl
    best_d2 = float("inf")
    best_idx = 0
    best_t = 0.0
    best_cx = pts[0][0]
    best_cy = pts[0][1]
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i+1]
        dx, dy = bx-ax, by-ay
        len_sq = dx*dx + dy*dy
        if len_sq == 0:
            t = 0.0
        else:
            t = ((px-ax)*dx + (py-ay)*dy) / len_sq
            t = max(0.0, min(1.0, t))
        cx = ax + t*dx
        cy = ay + t*dy
        ddx = px - cx; ddy = py - cy
        d2 = ddx*ddx + ddy*ddy
        if d2 < best_d2:
            best_d2 = d2; best_idx = i; best_t = t; best_cx = cx; best_cy = cy
    arc = cum[best_idx] + best_t * (cum[best_idx+1] - cum[best_idx])
    f = arc / total
    ax, ay = pts[best_idx]
    bx, by = pts[best_idx+1]
    tdx, tdy = bx-ax, by-ay
    tn = math.hypot(tdx, tdy) or 1
    tdx /= tn; tdy /= tn
    vx, vy = px - best_cx, py - best_cy
    cross = tdx*vy - tdy*vx
    dist = math.sqrt(best_d2)
    d = dist if cross >= 0 else -dist
    return f, d


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


def warp_events(events, src_cp, live_cp, tunnel_width, project_if_outside=True):
    cl_src = build_centerline(src_cp)
    cl_live = build_centerline(live_cp)
    out = []
    n_outside = 0
    n_projected = 0
    live_red = live_cp[-1]
    src_red = src_cp[-1]
    HIT_RADIUS = 18
    snap_from = -1
    for i, ev in enumerate(events):
        if math.hypot(ev["x"] - src_red["x"], ev["y"] - src_red["y"]) <= HIT_RADIUS:
            snap_from = i
            break
    for i, ev in enumerate(events):
        if snap_from >= 0 and i >= snap_from:
            wx = live_red["x"]
            wy = live_red["y"]
        else:
            f, d = project_to_centerline(ev["x"], ev["y"], cl_src)
            x, y, tx, ty = sample_at_fraction(f, cl_live)
            nx, ny = -ty, tx
            wx = x + d * nx
            wy = y + d * ny
            outside = abs(d) > tunnel_width
            if outside:
                n_outside += 1
                if project_if_outside:
                    margin = 2
                    shrunk = math.copysign(max(0, tunnel_width - margin), d)
                    wx = x + shrunk * nx
                    wy = y + shrunk * ny
                    n_projected += 1
        out.append({
            "x": wx, "y": wy,
            "timestamp": ev["timestamp"],
            "event_type": ev["event_type"],
        })
    return out, n_outside, n_projected, cl_live


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


def test_warp_math():
    print("\n=== Test 2: warp math ===")
    allowlist_path = os.path.join(REPO, "experiments_replay", "allowed_sessions_v1b.json")
    with open(allowlist_path) as f:
        allowlist = set(json.load(f))

    human_dir = os.path.join(REPO, "data", "human")

    # Group allowlisted completed humans by tunnel_id
    by_tid = {}
    for sid in allowlist:
        fpath = os.path.join(human_dir, f"{sid}.json")
        if not os.path.isfile(fpath): continue
        with open(fpath) as f:
            d = json.load(f)
        if not d.get("completed"): continue
        by_tid.setdefault(d["tunnel_id"], []).append(d)

    tids = sorted(by_tid.keys())
    assert len(tids) >= 2, "need at least 2 tunnels"
    src_tid = tids[0]
    live_tid = tids[1]
    src = by_tid[src_tid][0]
    live = by_tid[live_tid][0]
    tunnel_width = src.get("tunnel_width", 38)
    print(f"  source tunnel_id={src_tid} session={src['session_id'][:8]}  events={len(src['events'])}")
    print(f"  live   tunnel_id={live_tid} session={live['session_id'][:8]}  events={len(live['events'])}")
    print(f"  tunnel_width={tunnel_width}")

    src_cp = src["control_points"]
    live_cp = live["control_points"]
    warped, n_outside, n_projected, cl_live = warp_events(src["events"], src_cp, live_cp, tunnel_width)

    # (a) Endpoints near anchors
    live_start = live_cp[0]
    live_end = live_cp[-1]
    first = warped[0]
    last = warped[-1]
    d_start = math.hypot(first["x"] - live_start["x"], first["y"] - live_start["y"])
    d_end = math.hypot(last["x"] - live_end["x"], last["y"] - live_end["y"])
    print(f"  warped first point distance to live green anchor: {d_start:.2f} px")
    print(f"  warped last  point distance to live red   anchor: {d_end:.2f} px")

    # (b) Coordinate bounds
    xs = [e["x"] for e in warped]; ys = [e["y"] for e in warped]
    print(f"  warped x range: [{min(xs):.1f}, {max(xs):.1f}]   y range: [{min(ys):.1f}, {max(ys):.1f}]")
    assert -50 < min(xs) and max(xs) < 700, "warped x outside sane bounds"
    assert -50 < min(ys) and max(ys) < 400, "warped y outside sane bounds"

    # (c) Fraction inside live tunnel
    inside = 0
    for e in warped:
        d = dist_to_centerline(e["x"], e["y"], cl_live)
        if d <= tunnel_width:
            inside += 1
    frac_inside = inside / len(warped)
    print(f"  fraction of warped points within tunnel_width of live centerline: {frac_inside:.4f}")
    print(f"  n_outside={n_outside}  n_projected={n_projected}  total_events={len(warped)}")
    assert frac_inside >= 0.95, f"only {frac_inside:.4f} inside (<95%)"

    # Also pick a second pair for robustness
    print("  -- second pair --")
    src2_tid = tids[3] if len(tids) > 3 else tids[2]
    live2_tid = tids[4] if len(tids) > 4 else tids[0]
    src2 = by_tid[src2_tid][0]
    live2 = by_tid[live2_tid][0]
    warped2, n_out2, n_proj2, cl_live2 = warp_events(src2["events"], src2["control_points"], live2["control_points"], tunnel_width)
    inside2 = sum(1 for e in warped2 if dist_to_centerline(e["x"], e["y"], cl_live2) <= tunnel_width)
    frac2 = inside2 / len(warped2)
    print(f"  src tid={src2_tid} live tid={live2_tid}  frac inside={frac2:.4f}  n_outside={n_out2}  n_projected={n_proj2}")
    assert frac2 >= 0.95

    print("  Test 2 PASS")


if __name__ == "__main__":
    test_endpoint()
    test_warp_math()
    print("\nALL SMOKE TESTS PASSED")
