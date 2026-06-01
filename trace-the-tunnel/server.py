import argparse
import os
import json
import random
from flask import Flask, request, jsonify, send_from_directory

app = Flask(
    __name__,
    static_folder="static",
    template_folder=".",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TUNNEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "tunnel_config.json")

# Set by CLI flag at startup. When non-None, the server is in "experiment mode":
# every save is forced into data/<EXPERIMENT>/ and the source field is overwritten.
EXPERIMENT = None

# Optional whitelist for /api/human_bank. When non-None, only sessions whose
# session_id is in this set are returned. Used to enforce disjoint source/eval
# splits in the replay-v1b experiment.
ALLOWED_SESSIONS = None

# When True, game.js mounts window.__tunnelGame and prints the seed into the
# session display. Off by default so an in-page agent can't read the centerline
# straight from JS globals — they have to look at the canvas like a human does.
# Orchestration harnesses (experiments_replay_v2, experiments_warp, etc.) opt in
# via --expose-debug.
EXPOSE_DEBUG = False

# When True, /api/mode reports dynamic=True and the client generates a fresh
# random seed for every session instead of cycling through the fixed pool.
DYNAMIC = False

# ---------------------------------------------------------------------------
# Tunnel config: a fixed pool of seeds so humans and agents play the same set
# ---------------------------------------------------------------------------

def load_or_create_config(num_tunnels=10):
    """Load existing tunnel config or generate a new one with fixed seeds."""
    if os.path.exists(TUNNEL_CONFIG_PATH):
        with open(TUNNEL_CONFIG_PATH) as f:
            return json.load(f)

    rng = random.Random(42)  # deterministic generation
    tunnels = []
    for i in range(num_tunnels):
        tunnels.append({
            "tunnel_id": i,
            "seed": rng.randint(1, 2147483647),
        })

    config = {"tunnels": tunnels}
    with open(TUNNEL_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[config] Created {TUNNEL_CONFIG_PATH} with {num_tunnels} tunnels")
    return config

tunnel_config = load_or_create_config()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(".", "game.html")

@app.route("/api/tunnels", methods=["GET"])
def get_tunnels():
    """Return the full list of tunnel configs so the client can iterate them."""
    return jsonify(tunnel_config)

@app.route("/api/mode", methods=["GET"])
def get_mode():
    """Lets the harness verify the server is running with the expected experiment."""
    return jsonify({"experiment": EXPERIMENT, "expose_debug": EXPOSE_DEBUG, "dynamic": DYNAMIC})

@app.route("/api/human_bank/<int:tunnel_id>", methods=["GET"])
def get_human_bank(tunnel_id):
    """Return all completed human trajectories for a given tunnel_id.

    Used by the replay experiment so the agent can sample real human traces
    keyed to the live tunnel and dispatch them back into the canvas.
    """
    human_dir = os.path.join(DATA_DIR, "human")
    if not os.path.isdir(human_dir):
        return jsonify([])
    out = []
    for fname in sorted(os.listdir(human_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(human_dir, fname)) as f:
                d = json.load(f)
        except Exception:
            continue
        if not d.get("completed"):
            continue
        if d.get("tunnel_id") != tunnel_id:
            continue
        sid = d.get("session_id")
        if ALLOWED_SESSIONS is not None and sid not in ALLOWED_SESSIONS:
            continue
        out.append({
            "session_id": sid,
            "tunnel_id": d.get("tunnel_id"),
            "events": d.get("events", []),
        })
    return jsonify(out)

@app.route("/api/human_trace_full/<session_id>", methods=["GET"])
def get_human_trace_full(session_id):
    """Return a single completed human trajectory by session_id, including
    its control_points so the warp solver can reconstruct the source tunnel's
    centerline. Only sessions in the allowlist are returned (when set);
    otherwise any completed human trace can be fetched.

    Used by the warp experiment so the agent can sample one source trace
    and warp it onto a different live tunnel's centerline.
    """
    human_dir = os.path.join(DATA_DIR, "human")
    if not os.path.isdir(human_dir):
        return jsonify({"error": "no human data"}), 404
    if ALLOWED_SESSIONS is not None and session_id not in ALLOWED_SESSIONS:
        return jsonify({"error": "session not in allowlist"}), 404
    fpath = os.path.join(human_dir, f"{session_id}.json")
    if not os.path.isfile(fpath):
        return jsonify({"error": "session not found"}), 404
    try:
        with open(fpath) as f:
            d = json.load(f)
    except Exception as e:
        return jsonify({"error": f"failed to load: {e}"}), 500
    if not d.get("completed"):
        return jsonify({"error": "session not completed"}), 404
    return jsonify({
        "session_id": d.get("session_id"),
        "tunnel_id": d.get("tunnel_id"),
        "control_points": d.get("control_points", []),
        "tunnel_width": d.get("tunnel_width"),
        "canvas_size": d.get("canvas_size"),
        "events": d.get("events", []),
    })

@app.route("/api/save_trajectory", methods=["POST"])
def save_trajectory():
    data = request.json
    if not data or "session_id" not in data:
        return jsonify({"error": "missing session data"}), 400

    if EXPERIMENT is not None:
        # In experiment mode: refuse anything claiming to be human, then force
        # the source. The agent literally cannot route a save anywhere else.
        if data.get("source") == "human":
            return jsonify({"error": f"server in experiment mode '{EXPERIMENT}', refusing source='human'"}), 403
        source = EXPERIMENT
        data["source"] = EXPERIMENT
    else:
        source = data.get("source") or "unknown"

    sub_dir = os.path.join(DATA_DIR, source)
    os.makedirs(sub_dir, exist_ok=True)

    filename = f"{data['session_id']}.json"
    filepath = os.path.join(sub_dir, filename)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[saved] {source}/{filename}  tunnel_id={data.get('tunnel_id', '?')}  events={len(data.get('events', []))}")
    return jsonify({"status": "ok", "path": f"{source}/{filename}"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        default=None,
        help="If set, all saves are forced into data/<NAME>/ and source field is overwritten. "
             "Refuses source='human'. Leave unset for human collection.",
    )
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Generate a fresh random tunnel seed each session instead of cycling "
             "through the fixed pool. Intended for live deployment where tunnel "
             "reuse would let attackers pre-compute paths.",
    )
    parser.add_argument(
        "--expose-debug",
        action="store_true",
        help="Mount window.__tunnelGame on the page and print seed in the session "
             "display. Required by orchestration harnesses that read centerline/"
             "control_points or call loadTunnel from JS. Off by default so an "
             "in-page agent cannot bypass the canvas.",
    )
    parser.add_argument(
        "--allowed-sessions",
        default=None,
        help="Path to a JSON file containing a list of session_ids. When set, "
             "/api/human_bank only returns sessions whose id is in this list. "
             "Used to enforce a disjoint source/eval split for replay experiments.",
    )
    args = parser.parse_args()

    if args.experiment == "human":
        parser.error("--experiment cannot be 'human' (that folder is reserved for human data)")

    EXPERIMENT = args.experiment
    EXPOSE_DEBUG = args.expose_debug
    DYNAMIC = args.dynamic
    if DYNAMIC:
        print("[mode] --dynamic ON: each session gets a fresh random tunnel")
    if EXPOSE_DEBUG:
        print("[debug] --expose-debug ON: window.__tunnelGame is mounted on the page")

    if args.allowed_sessions:
        with open(args.allowed_sessions) as f:
            ALLOWED_SESSIONS = set(json.load(f))
        print(f"[bank] /api/human_bank restricted to {len(ALLOWED_SESSIONS)} session_ids from {args.allowed_sessions}")

    os.makedirs(os.path.join(DATA_DIR, "human"), exist_ok=True)
    if EXPERIMENT:
        os.makedirs(os.path.join(DATA_DIR, EXPERIMENT), exist_ok=True)
        print(f"[mode] experiment='{EXPERIMENT}' — saves forced to data/{EXPERIMENT}/")
    else:
        print(f"[mode] default — saves routed by client-supplied 'source' field")

    app.run(debug=True, port=args.port)
