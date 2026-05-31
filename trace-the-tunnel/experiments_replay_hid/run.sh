#!/usr/bin/env bash
# Convenience wrapper for the replay_hid experiment (Quartz CGEventPost).
#
# Usage:
#   ./experiments_replay_hid/run.sh [PER_TUNNEL]
#
# Server must be running in another terminal as:
#   python server.py --experiment replay_hid --expose-debug \
#     --allowed-sessions experiments_replay/allowed_sessions_v1b.json
#
# IMPORTANT: keep hands off the trackpad while a run is in progress. CGEvent
# injects directly into the macOS HID stream, so wiggling the real mouse
# while the script runs corrupts the trace. The browser must also stay
# foreground; the script calls bring_to_front before each dispatch but a
# system notification or alt-tab can steal focus mid-trace.
set -euo pipefail

PER_TUNNEL=${1:-8}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate

python experiments_replay_hid/replay_hid_solver.py --per-tunnel "$PER_TUNNEL" "${@:2}"
