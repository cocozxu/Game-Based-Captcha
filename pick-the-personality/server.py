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
PAD = 40
PERSONALITIES = ["leader", "follower", "hyper", "lazy"]
COLORS = ["#ef4444", "#22c55e", "#3b82f6", "#f59e0b"]


def make_round(seed: int):
    rng = random.Random(seed)
    slot_personalities = PERSONALITIES.copy()
    rng.shuffle(slot_personalities)
    target_personality = rng.choice(PERSONALITIES)
    target_slot = slot_personalities.index(target_personality)

    # All dots start in a tight row at the bottom of the canvas — leader will
    # head straight up and the rest will (visibly) follow.
    bottom_y = CANVAS_H - 80
    spacing = 55
    base_x = CANVAS_W / 2 - (3 * spacing) / 2
    dots = []
    for i in range(4):
        dots.append({
            "slot": i,
            "color": COLORS[i],
            "personality": slot_personalities[i],
            "start_x": base_x + i * spacing + (rng.random() - 0.5) * 10,
            "start_y": bottom_y + (rng.random() - 0.5) * 16,
            "behavior_seed": rng.randrange(2**31),
        })
    return {
        "seed": seed,
        "target_personality": target_personality,
        "target_slot": target_slot,
        "dots": dots,
    }


@app.route("/")
def index():
    return send_from_directory(ROOT, "game.html")


@app.route("/api/round", methods=["GET"])
def get_round():
    seed_arg = request.args.get("seed")
    seed = int(seed_arg) if seed_arg else random.randint(1, 2**31 - 1)
    r = make_round(seed)
    return jsonify({
        "seed": r["seed"],
        "target_personality": r["target_personality"],
        "dots": r["dots"],
    })


@app.route("/api/submit", methods=["POST"])
def submit():
    body = request.json or {}
    seed = body.get("seed")
    pick = body.get("pick")
    source = body.get("source", "unknown")
    if seed is None or pick is None:
        return jsonify({"error": "need seed and pick"}), 400

    r = make_round(int(seed))
    correct = (int(pick) == r["target_slot"])

    os.makedirs(os.path.join(DATA_DIR, source), exist_ok=True)
    rec = {
        "session_id": body.get("session_id", str(uuid.uuid4())),
        "ts": time.time(),
        "seed": int(seed),
        "pick": int(pick),
        "target_slot": r["target_slot"],
        "target_personality": r["target_personality"],
        "slot_personalities": [d["personality"] for d in r["dots"]],
        "correct": correct,
        "trajectory": body.get("trajectory", []),
        "source": source,
    }
    fname = f"{rec['session_id']}__{int(time.time()*1000)}.json"
    with open(os.path.join(DATA_DIR, source, fname), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[submit] source={source} seed={seed} pick={pick} "
          f"target={r['target_slot']} ({r['target_personality']}) "
          f"-> {'OK' if correct else 'WRONG'}")
    return jsonify({
        "correct": correct,
        "target_slot": r["target_slot"],
        "target_personality": r["target_personality"],
        "slot_personalities": [d["personality"] for d in r["dots"]],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5074)
    args = parser.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[mode] serving on http://localhost:{args.port}")
    app.run(debug=True, port=args.port)
