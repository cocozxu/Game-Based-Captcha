#!/usr/bin/env bash
# Convenience wrapper for the replay_cdp experiment.
#
# Usage:
#   ./experiments_replay_cdp/run.sh [PER_TUNNEL]
#
# Assumes the server is already running in another terminal as:
#   python server.py --experiment replay_cdp --expose-debug \
#     --allowed-sessions experiments_replay/allowed_sessions_v1b.json
set -euo pipefail

PER_TUNNEL=${1:-8}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate

python experiments_replay_cdp/replay_cdp_solver.py --per-tunnel "$PER_TUNNEL" "${@:2}"
