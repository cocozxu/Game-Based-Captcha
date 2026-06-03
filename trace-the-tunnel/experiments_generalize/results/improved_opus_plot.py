"""Generate side-by-side trajectory + speed plots comparing
human / original opus / improved opus on test tunnels 2, 6, 7.

Output: results/improved_opus_comparison.png
"""

import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "analysis"))
sys.path.insert(0, os.path.join(REPO, "experiments_generalize", "solvers", "split_a"))
import opus as opus_old  # noqa: E402

sys.path.remove(os.path.join(REPO, "experiments_generalize", "solvers", "split_a"))
sys.path.insert(0, os.path.join(REPO, "experiments_generalize", "solvers_improved", "split_a"))
import opus as opus_new  # noqa: E402


def reconstruct_cl(cps, n_per_seg=200):
    out = []
    for s in range(len(cps) // 4):
        p0, p1, p2, p3 = cps[s * 4 : s * 4 + 4]
        for i in range(n_per_seg):
            t = i / (n_per_seg - 1)
            u = 1 - t
            cx = u**3 * p0["x"] + 3 * u**2 * t * p1["x"] + 3 * u * t**2 * p2["x"] + t**3 * p3["x"]
            cy = u**3 * p0["y"] + 3 * u**2 * t * p1["y"] + 3 * u * t**2 * p2["y"] + t**3 * p3["y"]
            out.append((cx, cy))
    return np.array(out)


def best_human_for_tunnel(tid, n=1):
    """Return one human trace for the given tunnel (closest to median duration)."""
    cands = []
    for p in sorted(glob.glob(os.path.join(REPO, "data", "human", "*.json"))):
        j = json.load(open(p))
        if j.get("tunnel_id") != tid:
            continue
        moves = [e for e in j.get("events", []) if e["event_type"] == "mousemove"]
        if len(moves) < 5:
            continue
        t = np.array([e["timestamp"] for e in moves])
        cands.append((j, t[-1] - t[0]))
    if not cands:
        return None
    cands.sort(key=lambda x: x[1])
    return cands[len(cands) // 2][0]


def extract_xyt(j):
    moves = [e for e in j["events"] if e["event_type"] == "mousemove"]
    x = np.array([e["x"] for e in moves])
    y = np.array([e["y"] for e in moves])
    t = np.array([e["timestamp"] for e in moves])
    return x, y, t


def speed_curve(x, y, t):
    dt = np.diff(t)
    dt = np.where(dt == 0, 1e-3, dt)
    sp = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2) / dt
    return t[:-1] - t[0], sp * 1000.0  # px/s


def boundary_polygon(cl, half_w):
    """Approximate the tunnel boundary by offsetting the centerline ± half_w."""
    diffs = np.gradient(cl, axis=0)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    tang = diffs / norms
    perp = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    left = cl + perp * half_w
    right = cl - perp * half_w
    return left, right


def get_spec(tid):
    for p in sorted(glob.glob(os.path.join(REPO, "data", "human", "*.json"))):
        j = json.load(open(p))
        if j.get("tunnel_id") == tid:
            return {
                "tunnel_id": tid,
                "tunnel_seed": j.get("tunnel_seed", 0),
                "control_points": j.get("control_points", []),
                "tunnel_width": j.get("tunnel_width", 38),
                "canvas_size": j.get("canvas_size", {"width": 600, "height": 350}),
                "viewport": j.get("viewport", {"width": 600, "height": 350}),
            }
    return None


def main():
    tunnels = [2, 6, 7]
    seed = 7  # arbitrary seed for the attack traces shown

    fig, axes = plt.subplots(len(tunnels), 2, figsize=(13, 4 * len(tunnels)))
    if len(tunnels) == 1:
        axes = axes[None, :]

    for row, tid in enumerate(tunnels):
        spec = get_spec(tid)
        cl = reconstruct_cl(spec["control_points"])
        half_w = spec["tunnel_width"]
        left, right = boundary_polygon(cl, half_w)

        human = best_human_for_tunnel(tid)
        hx, hy, ht = extract_xyt(human)

        evts_old = opus_old.generate(spec, seed)
        ox, oy, ot = extract_xyt({"events": evts_old})

        evts_new = opus_new.generate(spec, seed)
        nx, ny, nt = extract_xyt({"events": evts_new})

        ax_traj = axes[row, 0]
        # Tunnel band as a shaded polygon
        poly_x = np.concatenate([left[:, 0], right[::-1, 0]])
        poly_y = np.concatenate([left[:, 1], right[::-1, 1]])
        ax_traj.fill(poly_x, poly_y, color="0.85", alpha=0.6, zorder=0)
        ax_traj.plot(cl[:, 0], cl[:, 1], "--", color="0.45", lw=0.8,
                     label="centerline", zorder=1)
        ax_traj.plot(hx, hy, color="tab:green", lw=1.6, label="human", zorder=2)
        ax_traj.plot(ox, oy, color="tab:blue", lw=1.4, label="opus (orig)",
                     alpha=0.85, zorder=3)
        ax_traj.plot(nx, ny, color="tab:red", lw=1.4, label="opus (improved)",
                     alpha=0.85, zorder=4)
        ax_traj.invert_yaxis()  # canvas y goes down
        ax_traj.set_aspect("equal")
        ax_traj.set_xlim(-20, 620)
        ax_traj.set_ylim(370, -20)
        ax_traj.set_title(f"tunnel {tid} — trajectory")
        ax_traj.legend(loc="lower right", fontsize=8)
        ax_traj.set_xlabel("x (px)")
        ax_traj.set_ylabel("y (px)")

        ax_sp = axes[row, 1]
        for label, x, y, t, color in [
            ("human", hx, hy, ht, "tab:green"),
            ("opus (orig)", ox, oy, ot, "tab:blue"),
            ("opus (improved)", nx, ny, nt, "tab:red"),
        ]:
            tt, sp = speed_curve(x, y, t)
            ax_sp.plot(tt, sp, label=label, color=color, lw=1.0, alpha=0.9)
        ax_sp.set_title(f"tunnel {tid} — speed vs time")
        ax_sp.set_xlabel("t (ms)")
        ax_sp.set_ylabel("speed (px/s)")
        ax_sp.set_ylim(0, 1800)
        ax_sp.legend(loc="upper right", fontsize=8)
        ax_sp.grid(alpha=0.3)

    fig.suptitle(
        "Improved opus (lazy path + lognormal submovements) vs original opus, "
        "test tunnels 2/6/7 (split_a)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(HERE, "improved_opus_comparison.png")
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
