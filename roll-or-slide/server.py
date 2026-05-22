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


def make_round_v5(seed: int):
    rng = random.Random(seed)
    anomaly_slot = rng.randrange(4)
    anomaly_motion = ANOMALY_MOTIONS_V4[rng.randrange(len(ANOMALY_MOTIONS_V4))]
    shapes = []
    for i in range(4):
        motion = anomaly_motion if i == anomaly_slot else "linear"
        keyframes = []
        for step in range(N_STEPS_V4):
            noise = (rng.random() - 0.5) * 0.02
            pos = position_v4(0.0, 1.0, step, N_STEPS_V4, motion, noise)
            keyframes.append(round(max(0.0, min(1.0, pos)), 4))
        shapes.append({"slot": i, "keyframes": keyframes})
    return {
        "seed": seed,
        "anomaly_slot": anomaly_slot,
        "anomaly_motion": anomaly_motion,
        "shapes": shapes,
    }


ANOMALY_TYPES_V6 = ["floater", "heavy", "hyper", "drifter"]

def make_round_v6(seed: int):
    rng = random.Random(seed)
    anomaly_slot = rng.randrange(4)
    anomaly_type = ANOMALY_TYPES_V6[rng.randrange(len(ANOMALY_TYPES_V6))]
    shapes = []
    for i in range(4):
        x0 = 0.15 + rng.random() * 0.70
        y0 = 0.10 + rng.random() * 0.35
        vx0 = (rng.random() - 0.5) * 280
        vy0 = rng.random() * 80
        if i == anomaly_slot:
            if anomaly_type == "floater":
                grav, ela, drift = 0.0, 0.72, 0.0
            elif anomaly_type == "heavy":
                grav, ela, drift = 3.0, 0.22, 0.0
            elif anomaly_type == "hyper":
                grav, ela, drift = 1.0, 0.96, 0.0
            else:  # drifter
                grav, ela = 1.0, 0.68
                drift = 200.0 if rng.random() < 0.5 else -200.0
        else:
            grav, ela, drift = 1.0, 0.68, 0.0
        shapes.append({
            "slot": i,
            "x0": round(x0, 4), "y0": round(y0, 4),
            "vx0": round(vx0, 2), "vy0": round(vy0, 2),
            "grav": grav, "ela": ela, "drift": drift,
        })
    return {"seed": seed, "anomaly_slot": anomaly_slot,
            "anomaly_type": anomaly_type, "shapes": shapes}


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


@app.route("/v5")
def index_v5():
    return send_from_directory(ROOT, "game_v5.html")


@app.route("/v6")
def index_v6():
    return send_from_directory(ROOT, "game_v6.html")


@app.route("/api/round_v6", methods=["GET"])
def get_round_v6():
    seed_arg = request.args.get("seed")
    seed = int(seed_arg) if seed_arg else random.randint(1, 2**31 - 1)
    r = make_round_v6(seed)
    return jsonify({"seed": r["seed"], "shapes": r["shapes"]})


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


@app.route("/api/round_v5", methods=["GET"])
def get_round_v5():
    seed_arg = request.args.get("seed")
    seed = int(seed_arg) if seed_arg else random.randint(1, 2**31 - 1)
    r = make_round_v5(seed)
    # Return keyframes only — no anomaly_slot, no motion types
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

    if version == "v6":
        r = make_round_v6(int(seed))
        correct = (int(pick) == r["anomaly_slot"])
        os.makedirs(os.path.join(DATA_DIR, source), exist_ok=True)
        rec = {
            "session_id": body.get("session_id", str(uuid.uuid4())),
            "ts": time.time(),
            "seed": int(seed),
            "pick": int(pick),
            "anomaly_slot": r["anomaly_slot"],
            "anomaly_type": r["anomaly_type"],
            "correct": correct,
            "events": body.get("events", []),
            "mouse_moves": body.get("mouse_moves", []),
            "source": source,
            "version": version,
        }
        fname = f"{rec['session_id']}__{int(time.time()*1000)}.json"
        with open(os.path.join(DATA_DIR, source, fname), "w") as f:
            json.dump(rec, f, indent=2)
        elapsed_ms = next((e.get("elapsed_ms") for e in reversed(rec["events"])
                           if e.get("type") == "click"), None)
        print(f"[submit] source={source} version=v6 seed={seed} pick={pick} "
              f"anomaly={r['anomaly_slot']} type={r['anomaly_type']} "
              f"elapsed_ms={elapsed_ms} -> {'OK' if correct else 'WRONG'}")
        return jsonify({"correct": correct, "anomaly_slot": r["anomaly_slot"],
                        "anomaly_type": r["anomaly_type"]})

    if version == "v5":
        r = make_round_v5(int(seed))
        correct = (int(pick) == r["anomaly_slot"])
        os.makedirs(os.path.join(DATA_DIR, source), exist_ok=True)
        rec = {
            "session_id": body.get("session_id", str(uuid.uuid4())),
            "ts": time.time(),
            "seed": int(seed),
            "pick": int(pick),
            "anomaly_slot": r["anomaly_slot"],
            "anomaly_motion": r["anomaly_motion"],
            "correct": correct,
            "trajectory": body.get("trajectory", []),
            "hover_log": body.get("hover_log", []),
            "mouse_moves": body.get("mouse_moves", []),
            "source": source,
            "version": version,
        }
        fname = f"{rec['session_id']}__{int(time.time()*1000)}.json"
        with open(os.path.join(DATA_DIR, source, fname), "w") as f:
            json.dump(rec, f, indent=2)
        elapsed_ms = next((e.get("elapsed_ms") for e in reversed(rec["trajectory"]) if e.get("type") == "pick"), None)
        cycles = next((e.get("cycle") for e in reversed(rec["trajectory"]) if e.get("type") == "pick"), None)
        print(f"[submit] source={source} version=v5 seed={seed} pick={pick} anomaly={r['anomaly_slot']} "
              f"motion={r['anomaly_motion']} elapsed_ms={elapsed_ms} cycles={cycles} -> {'OK' if correct else 'WRONG'}")
        return jsonify({"correct": correct, "anomaly_slot": r["anomaly_slot"]})

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
