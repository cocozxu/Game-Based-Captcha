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
    return jsonify({"experiment": EXPERIMENT})

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
        source = data.get("source", "unknown")

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
    args = parser.parse_args()

    if args.experiment == "human":
        parser.error("--experiment cannot be 'human' (that folder is reserved for human data)")

    EXPERIMENT = args.experiment

    os.makedirs(os.path.join(DATA_DIR, "human"), exist_ok=True)
    if EXPERIMENT:
        os.makedirs(os.path.join(DATA_DIR, EXPERIMENT), exist_ok=True)
        print(f"[mode] experiment='{EXPERIMENT}' — saves forced to data/{EXPERIMENT}/")
    else:
        print(f"[mode] default — saves routed by client-supplied 'source' field")

    app.run(debug=True, port=args.port)
