"""
render_prompt.py — interpolate splits.yaml + on-disk human traces into
one ready-to-paste prompt per split.

For each split in splits.yaml, emit:
  prompts/rendered/{split_id}.md

Inputs to the prompt for a split:
  - demo tunnel geometries (control_points, tunnel_width, canvas_size, viewport)
  - N example human traces per demo tunnel (only `completed: true` traces)
  - test tunnel geometries (no traces)

Reads directly from data/human/*.json — no server required.
"""

import argparse
import glob
import json
import os
import random
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
HUMAN_DIR = os.path.join(REPO, "data", "human")
TUNNEL_IDS = list(range(10))


def load_human_traces_by_tunnel():
    """Read every data/human/*.json and group by tunnel_id."""
    by_tid = {tid: [] for tid in TUNNEL_IDS}
    for path in glob.glob(os.path.join(HUMAN_DIR, "*.json")):
        try:
            with open(path) as f:
                trace = json.load(f)
        except Exception as e:
            print(f"  WARN: failed to read {path}: {e}", file=sys.stderr)
            continue
        tid = trace.get("tunnel_id")
        if tid in by_tid:
            by_tid[tid].append(trace)
    return by_tid


def tunnel_spec_from_trace(trace):
    return {
        "tunnel_id": trace["tunnel_id"],
        "tunnel_seed": trace["tunnel_seed"],
        "control_points": trace["control_points"],
        "tunnel_width": trace["tunnel_width"],
        "canvas_size": trace["canvas_size"],
        "viewport": trace["viewport"],
    }


def event_stream(trace):
    return [
        {
            "x": e["x"],
            "y": e["y"],
            "timestamp": e["timestamp"],
            "event_type": e["event_type"],
            "inside_tunnel": e["inside_tunnel"],
        }
        for e in trace["events"]
    ]


def render_split(split, by_tid, n_examples, traces_per_test, template):
    rng = random.Random(hash(split["id"]) & 0xFFFFFFFF)

    demo_geometries = {}
    demo_traces_by_tunnel = {}
    for tid in split["demo"]:
        traces = by_tid[tid]
        if not traces:
            raise RuntimeError(f"no human traces on disk for demo tunnel {tid}")
        demo_geometries[str(tid)] = tunnel_spec_from_trace(traces[0])
        completed = [t for t in traces if t.get("completed", False)]
        pool = completed if completed else traces
        chosen = rng.sample(pool, min(n_examples, len(pool)))
        demo_traces_by_tunnel[str(tid)] = [event_stream(t) for t in chosen]

    test_geometries = {}
    for tid in split["test"]:
        traces = by_tid[tid]
        if not traces:
            raise RuntimeError(f"no human traces on disk for test tunnel {tid}")
        test_geometries[str(tid)] = tunnel_spec_from_trace(traces[0])

    rendered = template
    rendered = rendered.replace("{{n_examples_per_tunnel}}", str(n_examples))
    rendered = rendered.replace("{{n_demo_tunnels}}", str(len(split["demo"])))
    rendered = rendered.replace("{{split_id}}", split["id"])
    rendered = rendered.replace("{{traces_per_test_tunnel}}", str(traces_per_test))
    rendered = rendered.replace(
        "{{demo_geometries_json}}",
        f"```json\n{json.dumps(demo_geometries, indent=2)}\n```",
    )
    rendered = rendered.replace(
        "{{demo_traces_json}}",
        f"```json\n{json.dumps(demo_traces_by_tunnel)}\n```",
    )
    rendered = rendered.replace(
        "{{test_geometries_json}}",
        f"```json\n{json.dumps(test_geometries, indent=2)}\n```",
    )
    return rendered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=os.path.join(HERE, "splits.yaml"))
    ap.add_argument("--template", default=os.path.join(HERE, "prompts", "solver_prompt.tmpl"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "prompts", "rendered"))
    ap.add_argument("--n-examples", type=int, default=3,
                    help="number of human trace examples per demo tunnel to include")
    args = ap.parse_args()

    with open(args.splits) as f:
        cfg = yaml.safe_load(f)
    with open(args.template) as f:
        template = f.read()

    print(f"Loading human traces from {HUMAN_DIR} ...")
    by_tid = load_human_traces_by_tunnel()
    for tid in TUNNEL_IDS:
        n_total = len(by_tid[tid])
        n_complete = sum(1 for t in by_tid[tid] if t.get("completed", False))
        print(f"  tunnel {tid}: {n_total} traces  ({n_complete} completed)")

    os.makedirs(args.out_dir, exist_ok=True)
    traces_per_test = cfg["generation"]["traces_per_test_tunnel"]
    for split in cfg["splits"]:
        rendered = render_split(split, by_tid, args.n_examples, traces_per_test, template)
        out_path = os.path.join(args.out_dir, f"{split['id']}.md")
        with open(out_path, "w") as f:
            f.write(rendered)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  wrote {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
