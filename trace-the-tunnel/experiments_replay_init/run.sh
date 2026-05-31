#!/usr/bin/env bash
# Convenience wrapper for the replay_init experiment (Web API forge via
# CDP addInitScript). Headless OK — the forge bypasses the OS pipeline
# entirely.
#
# Server must be running in another terminal as:
#   python server.py --experiment replay_init --expose-debug \
#     --allowed-sessions experiments_replay/allowed_sessions_v1b.json
set -euo pipefail

PER_TUNNEL=${1:-8}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate

python experiments_replay_init/replay_init_solver.py --per-tunnel "$PER_TUNNEL" "${@:2}"
