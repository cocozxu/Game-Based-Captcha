"""
Pre-select a small set of completed human trajectories to act as the
disjoint source bank for a replay experiment.

Two selection modes:

  --per-tunnel N
      For every tunnel_id that has at least N completed humans, pick
      N at random. Use this for v1b: full tunnel coverage with a
      constrained bank size per tunnel.

  --total N
      Pick N completed humans total, stratified across distinct
      tunnels to maximize tunnel coverage. Use this for v1c: probe
      how small the bank can get before the attack stops working.

  --only-tunnel TID --count N
      Pick N completed humans from a single tunnel_id (TID). Use
      this to test single-tunnel-observability attacks: the warp
      solver's cross-tunnel constraint will then skip TID itself
      and warp the TID humans onto all other tunnels.

Outputs (relative to trace-the-tunnel/):

  experiments_replay/<out>.json
      Flat JSON list of selected session_ids. The server loads this
      via --allowed-sessions to filter /api/human_bank.

  experiments_replay/<out>_meta.json
      Human-readable manifest: seed, mode, per-tunnel counts, source
      filenames, event counts. Source of truth for "what was the
      bank for this run."

Usage:

  python experiments_replay/select_bank.py --per-tunnel 3 --seed 42 \
      --out experiments_replay/allowed_sessions_v1b.json

  python experiments_replay/select_bank.py --total 5 --seed 42 \
      --out experiments_replay/allowed_sessions_v1c.json
"""

import argparse
import datetime
import json
import os
import random
import sys
from collections import defaultdict


def load_completed_humans(human_dir):
    rows = []
    for fname in sorted(os.listdir(human_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(human_dir, fname)) as f:
            d = json.load(f)
        if not d.get("completed"):
            continue
        rows.append({
            "fname": fname,
            "session_id": d.get("session_id"),
            "tunnel_id": d.get("tunnel_id"),
            "n_events": len(d.get("events", [])),
            "duration_ms": d.get("duration_ms"),
        })
    return rows


def select_per_tunnel(rows, n, rng):
    by_tid = defaultdict(list)
    for r in rows:
        by_tid[r["tunnel_id"]].append(r)
    selected = []
    skipped = []
    for tid in sorted(by_tid.keys()):
        pool = by_tid[tid]
        if len(pool) < n:
            skipped.append({"tunnel_id": tid, "available": len(pool), "needed": n})
            continue
        picks = rng.sample(pool, n)
        selected.extend(picks)
    return selected, skipped


def select_only_tunnel(rows, tid, n, rng):
    pool = [r for r in rows if r["tunnel_id"] == tid]
    if len(pool) < n:
        return [], [{"tunnel_id": tid, "available": len(pool), "needed": n}]
    return rng.sample(pool, n), []


def select_total(rows, n, rng):
    """Pick n sessions, biased toward distinct tunnels.

    Strategy: shuffle tunnels, draw one from each in turn until we've
    selected n. With n <= num_tunnels this guarantees all-different
    tunnels; with n > num_tunnels we wrap and start picking a second
    from each, still avoiding repeated session_ids.
    """
    by_tid = defaultdict(list)
    for r in rows:
        by_tid[r["tunnel_id"]].append(r)
    tids = sorted(by_tid.keys())
    rng.shuffle(tids)
    # Pre-shuffle each tunnel's pool
    for tid in tids:
        rng.shuffle(by_tid[tid])
    cursors = {tid: 0 for tid in tids}
    selected = []
    while len(selected) < n:
        progress = False
        for tid in tids:
            if len(selected) >= n:
                break
            if cursors[tid] < len(by_tid[tid]):
                selected.append(by_tid[tid][cursors[tid]])
                cursors[tid] += 1
                progress = True
        if not progress:
            break  # ran out of unique humans entirely
    return selected, []


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--per-tunnel", type=int, help="Pick N completed humans for every tunnel_id.")
    mode.add_argument("--total", type=int, help="Pick N completed humans total, stratified for tunnel coverage.")
    mode.add_argument("--only-tunnel", type=int, help="Pick humans only from this tunnel_id (requires --count).")
    parser.add_argument("--count", type=int, default=None, help="Number of humans to pick (with --only-tunnel).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="Output JSON file for the session_id list.")
    parser.add_argument("--human-dir", default=None, help="Override path to data/human/.")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, ".."))
    human_dir = args.human_dir or os.path.join(repo, "data", "human")

    if not os.path.isdir(human_dir):
        print(f"ERROR: human dir not found: {human_dir}", file=sys.stderr)
        sys.exit(1)

    rows = load_completed_humans(human_dir)
    print(f"Loaded {len(rows)} completed human trajectories from {human_dir}")

    rng = random.Random(args.seed)
    if args.per_tunnel is not None:
        mode_name = f"per-tunnel:{args.per_tunnel}"
        selected, skipped = select_per_tunnel(rows, args.per_tunnel, rng)
    elif args.only_tunnel is not None:
        if args.count is None:
            print("ERROR: --only-tunnel requires --count", file=sys.stderr)
            sys.exit(1)
        mode_name = f"only-tunnel:{args.only_tunnel}:{args.count}"
        selected, skipped = select_only_tunnel(rows, args.only_tunnel, args.count, rng)
    else:
        mode_name = f"total:{args.total}"
        selected, skipped = select_total(rows, args.total, rng)

    # Write the flat allowlist (what the server consumes)
    out_path = os.path.abspath(args.out)
    session_ids = [r["session_id"] for r in selected]
    with open(out_path, "w") as f:
        json.dump(session_ids, f, indent=2)

    # Write the per-run metadata next to it
    meta_path = out_path.replace(".json", "_meta.json")
    if meta_path == out_path:
        meta_path = out_path + ".meta.json"
    per_tid = defaultdict(list)
    for r in selected:
        per_tid[r["tunnel_id"]].append(r)
    meta = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "mode": mode_name,
        "n_pool": len(rows),
        "n_selected": len(selected),
        "tunnel_coverage": {str(tid): len(items) for tid, items in sorted(per_tid.items())},
        "skipped_tunnels": skipped,
        "selected": [
            {
                "session_id": r["session_id"],
                "tunnel_id": r["tunnel_id"],
                "fname": r["fname"],
                "n_events": r["n_events"],
                "duration_ms": r["duration_ms"],
            }
            for r in sorted(selected, key=lambda r: (r["tunnel_id"], r["session_id"]))
        ],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Selected {len(selected)} sessions ({mode_name})")
    print(f"  Tunnel coverage: {dict(meta['tunnel_coverage'])}")
    if skipped:
        print(f"  Skipped tunnels (insufficient humans): {skipped}")
    print(f"  Wrote allowlist: {out_path}")
    print(f"  Wrote metadata:  {meta_path}")


if __name__ == "__main__":
    main()
