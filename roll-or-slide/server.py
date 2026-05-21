import argparse
import json
import math
import os
import random
import time
import uuid
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder=None)

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")


CANVAS_W, CANVAS_H = 700, 600
TILE_W, TILE_H = CANVAS_W / 2, CANVAS_H / 2
N_VERTS = 8
BASE_R = 38.0
JITTER = 0.45
# Per-shape multiplicative jitter on the slide amplitude, in [1 ± AMP_JITTER].
# At ±5% the variation is below the human perceptual threshold for amplitude
# but means no two trajectories match exactly — agents can't define one
# canonical slider trajectory and pick out the outlier as the residual.
AMP_JITTER = 0.05


def tile_bounds(i: int):
    col, row = i % 2, i // 2
    left, top = col * TILE_W, row * TILE_H
    return {
        "left": left,
        "top": top,
        "cx": left + TILE_W / 2,
        "cy": top + TILE_H / 2,
    }


def make_shape_verts(rng: random.Random):
    verts = []
    for k in range(N_VERTS):
        a = (k + (rng.random() - 0.5) * 0.3) / N_VERTS * 2 * math.pi
        r = BASE_R * (1 - JITTER / 2 + rng.random() * JITTER)
        verts.append([r * math.cos(a), r * math.sin(a)])
    return verts


SQUARE_HALF = BASE_R
SQUARE_VERTS = [
    [-SQUARE_HALF, -SQUARE_HALF],
    [ SQUARE_HALF, -SQUARE_HALF],
    [ SQUARE_HALF,  SQUARE_HALF],
    [-SQUARE_HALF,  SQUARE_HALF],
]
SQUARE_EFF_R = SQUARE_HALF * math.sqrt(2)

CIRCLE_R = BASE_R


def make_round(seed: int):
    rng = random.Random(seed)
    roller_slot = rng.randrange(4)
    round_axis = rng.random() * 2 * math.pi
    shapes = []
    for i in range(4):
        verts = make_shape_verts(rng)
        eff_r = sum(math.hypot(x, y) for x, y in verts) / len(verts)
        t = tile_bounds(i)
        shapes.append({
            "slot": i,
            "verts": verts,
            "eff_r": eff_r,
            "rest_x": t["cx"],
            "rest_y": t["cy"],
            "dir_angle": round_axis + (0.0 if rng.random() < 0.5 else math.pi),
            "amp_factor": 1.0 + (rng.random() * 2 - 1) * AMP_JITTER,
            "motion_type": "roll" if i == roller_slot else "slide",
        })
    return {"seed": seed, "roller_slot": roller_slot, "shapes": shapes}


def make_round_v2(seed: int):
    rng = random.Random(seed)
    roller_slot = rng.randrange(4)
    round_axis = rng.random() * 2 * math.pi
    shapes = []
    for i in range(4):
        t = tile_bounds(i)
        shapes.append({
            "slot": i,
            "verts": SQUARE_VERTS,
            "eff_r": SQUARE_EFF_R,
            "rest_x": t["cx"],
            "rest_y": t["cy"],
            "dir_angle": round_axis + (0.0 if rng.random() < 0.5 else math.pi),
            "amp_factor": 1.0 + (rng.random() * 2 - 1) * AMP_JITTER,
            "motion_type": "roll" if i == roller_slot else "slide",
        })
    return {"seed": seed, "roller_slot": roller_slot, "shapes": shapes}


def make_round_v3(seed: int):
    rng = random.Random(seed)
    roller_slot = rng.randrange(4)
    round_axis = rng.random() * 2 * math.pi
    shapes = []
    for i in range(4):
        t = tile_bounds(i)
        shapes.append({
            "slot": i,
            "radius": CIRCLE_R,
            "eff_r": CIRCLE_R,
            "rest_x": t["cx"],
            "rest_y": t["cy"],
            "dir_angle": round_axis + (0.0 if rng.random() < 0.5 else math.pi),
            "amp_factor": 1.0 + (rng.random() * 2 - 1) * AMP_JITTER,
            "motion_type": "roll" if i == roller_slot else "slide",
        })
    return {"seed": seed, "roller_slot": roller_slot, "shapes": shapes}


@app.route("/")
def index():
    return send_from_directory(ROOT, "game.html")


@app.route("/v2")
def index_v2():
    return send_from_directory(ROOT, "game_v2.html")


@app.route("/v3")
def index_v3():
    return send_from_directory(ROOT, "game_v3.html")


@app.route("/api/round", methods=["GET"])
def get_round():
    seed_arg = request.args.get("seed")
    seed = int(seed_arg) if seed_arg else random.randint(1, 2**31 - 1)
    r = make_round(seed)
    return jsonify({"seed": r["seed"], "shapes": r["shapes"]})


@app.route("/api/round_v2", methods=["GET"])
def get_round_v2():
    seed_arg = request.args.get("seed")
    seed = int(seed_arg) if seed_arg else random.randint(1, 2**31 - 1)
    r = make_round_v2(seed)
    return jsonify({"seed": r["seed"], "shapes": r["shapes"]})


@app.route("/api/round_v3", methods=["GET"])
def get_round_v3():
    seed_arg = request.args.get("seed")
    seed = int(seed_arg) if seed_arg else random.randint(1, 2**31 - 1)
    r = make_round_v3(seed)
    return jsonify({"seed": r["seed"], "shapes": r["shapes"]})


@app.route("/api/submit", methods=["POST"])
def submit():
    body = request.json or {}
    seed = body.get("seed")
    pick = body.get("pick")
    source = body.get("source", "unknown")
    version = body.get("version", "v1")
    if seed is None or pick is None:
        return jsonify({"error": "need seed and pick"}), 400

    if version == "v3":
        r = make_round_v3(int(seed))
    elif version == "v2":
        r = make_round_v2(int(seed))
    else:
        r = make_round(int(seed))
    correct = (int(pick) == r["roller_slot"])

    os.makedirs(os.path.join(DATA_DIR, source), exist_ok=True)
    rec = {
        "session_id": body.get("session_id", str(uuid.uuid4())),
        "ts": time.time(),
        "seed": int(seed),
        "pick": int(pick),
        "roller_slot": r["roller_slot"],
        "correct": correct,
        "trajectory": body.get("trajectory", []),
        "source": source,
        "version": version,
    }
    fname = f"{rec['session_id']}__{int(time.time()*1000)}.json"
    with open(os.path.join(DATA_DIR, source, fname), "w") as f:
        json.dump(rec, f, indent=2)

    n_previews = sum(1 for e in rec["trajectory"] if e.get("type") == "preview")
    print(f"[submit] source={source} version={version} seed={seed} pick={pick} roller={r['roller_slot']} "
          f"previews={n_previews} -> {'OK' if correct else 'WRONG'}")
    return jsonify({"correct": correct, "roller_slot": r["roller_slot"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5072)
    args = parser.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[mode] serving on http://localhost:{args.port}")
    app.run(debug=True, port=args.port)
