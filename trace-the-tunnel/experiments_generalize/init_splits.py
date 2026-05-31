"""
init_splits.py — random tunnel split sampler.

Writes splits.yaml with N splits, each picking K demo tunnels at random
from the 10 available (0-9). Test tunnels = complement of demo.

Hand-edit splits.yaml after if you want specific tunnels.
"""

import argparse
import os
import random

import yaml


HERE = os.path.dirname(os.path.abspath(__file__))
TUNNEL_IDS = list(range(10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="number of splits to sample")
    ap.add_argument("--demo-size", type=int, default=3, help="how many tunnels in demo set per split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--traces-per-test-tunnel", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "splits.yaml"))
    ap.add_argument("--force", action="store_true", help="overwrite existing splits.yaml")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"{args.out} exists. Pass --force to overwrite.")

    rng = random.Random(args.seed)
    splits = []
    seen = set()
    while len(splits) < args.n:
        demo = sorted(rng.sample(TUNNEL_IDS, args.demo_size))
        key = tuple(demo)
        if key in seen:
            continue
        seen.add(key)
        test = [t for t in TUNNEL_IDS if t not in demo]
        splits.append({
            "id": f"split_{chr(ord('a') + len(splits))}",
            "demo": demo,
            "test": test,
        })

    doc = {
        "generation": {
            "traces_per_test_tunnel": args.traces_per_test_tunnel,
            # Model identifiers used as filenames under solvers/<split_id>/.
            # The names are arbitrary labels — you decide which actual model
            # version to paste the prompt into.
            "models": ["opus", "sonnet", "haiku"],
        },
        "splits": splits,
    }
    with open(args.out, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    print(f"Wrote {args.out}")
    for s in splits:
        print(f"  {s['id']}: demo={s['demo']}  test={s['test']}")


if __name__ == "__main__":
    main()
