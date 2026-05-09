#!/bin/bash
# Collect agent trajectory data by running Claude Code once per ROUND.
# Each round plays all 10 tunnels in a single context, then resets.
#
# Prerequisites:
#   1. Server running: cd trace-the-tunnel && python server.py
#   2. Claude Code CLI available as `claude`
#   3. For --bc mode: trained model (python analysis/bc_agent.py --train)
#
# Usage:
#   ./collect_agent_data.sh                     # 1 round, naive agent
#   ./collect_agent_data.sh --rounds 3          # 3 rounds
#   ./collect_agent_data.sh --human-like        # use human_cursor.py
#   ./collect_agent_data.sh --bc               # use BC agent (LSTM trained on human data)
#   ./collect_agent_data.sh --bc --rounds 3    # 3 rounds of BC agent

set -e
cd "$(dirname "$0")"

# --- Parse flags ---
ROUNDS=1
MODE="naive"

while [[ $# -gt 0 ]]; do
  case $1 in
    --rounds)
      ROUNDS="$2"
      shift 2
      ;;
    --human-like)
      MODE="human-like"
      shift
      ;;
    --bc)
      MODE="bc"
      shift
      ;;
    *)
      echo "Unknown flag: $1"
      echo "Usage: $0 [--rounds N] [--human-like | --bc]"
      exit 1
      ;;
  esac
done

case $MODE in
  naive)
    SOURCE_LABEL="agent"
    MODE_EXTRA=""
    ;;
  human-like)
    SOURCE_LABEL="agent_humanlike"
    MODE_EXTRA="You have access to human_cursor.py (see CLAUDE.md for usage). Use it if you think it helps."
    ;;
  bc)
    SOURCE_LABEL="agent_bc"
    # BC mode: generate waypoints with the trained model, then replay via Playwright
    echo "=== BC Agent Mode ==="
    echo "Generating waypoints with trained LSTM model..."
    source .venv/bin/activate
    python analysis/bc_agent.py --play-all
    echo ""
    MODE_EXTRA="You are replaying PRE-COMPUTED waypoints from a trained behavioral cloning model.

For each tunnel, the waypoints are in data/bc_waypoints/tunnel_TUNNEL_ID.json.
Each file contains a JSON array of {x, y, delay_ms} objects in canvas coordinates.

To replay:
1. Read the waypoints file with the Read tool
2. Use browser_run_code_unsafe to replay them as mouse movements:
   - Get canvas rect for coordinate conversion
   - Mouse down on first waypoint, move through all with their delays, mouse up at end

Do NOT compute your own path. Use the pre-computed waypoints exactly."
    ;;
esac

mkdir -p data/logs

echo "=== Collecting agent trajectories ==="
echo "Mode: $MODE | Rounds: $ROUNDS"
echo ""

for round in $(seq 1 "$ROUNDS"); do
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  LOGFILE="data/logs/round_${round}_${MODE}_${TIMESTAMP}.txt"

  echo "=== Round $round of $ROUNDS ($MODE) — log: $LOGFILE ==="

  claude -p "You are playing the trace-the-tunnel game to collect agent trajectory data.
The game server is running at http://localhost:5050.
Read the CLAUDE.md file in this directory for reference on how the game works and what tools are available.

Your task: Play ALL tunnels 0 through 9, one at a time. Save each with source='$SOURCE_LABEL'.

For EACH tunnel (0, 1, 2, ... 9):
1. Navigate to http://localhost:5050 (first time only)
2. Load the tunnel: use browser_evaluate to call window.__tunnelGame.loadTunnel(TUNNEL_ID)
3. Trace through the tunnel. Try to move the way a real human would — think about how a person drags a cursor through a winding path. It's up to you how to achieve this.
4. Check the game state. If it failed (out of bounds), the game auto-resets the same tunnel. You MUST retry until you succeed. Adjust your approach — use more points, less noise, stay closer to centerline, etc.
5. Only after success ('done_success'), re-save with source='$SOURCE_LABEL' and move on to the next tunnel.
   Use browser_evaluate:
   (async () => {
     const data = window.__tunnelGame.getSessionData();
     data.source = '$SOURCE_LABEL';
     const resp = await fetch('/api/save_trajectory', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify(data),
     });
     return { ok: resp.ok, session_id: data.session_id, tunnel_id: data.tunnel_id, events: data.events.length, completed: data.completed };
   })()

$MODE_EXTRA

Start now. Navigate to the game and play all 10 tunnels." --verbose --allowedTools "mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_evaluate,mcp__plugin_playwright_playwright__browser_run_code_unsafe,mcp__plugin_playwright_playwright__browser_click,mcp__plugin_playwright_playwright__browser_snapshot,Read,Bash" 2>&1 | tee "$LOGFILE"

  echo ""
  echo "=== Round $round done — log saved to $LOGFILE ==="
  echo ""
done

echo "=== All done! ==="
echo "Trajectories: data/$SOURCE_LABEL/"
echo "Agent logs:   data/logs/"
ls -la "data/$SOURCE_LABEL/" 2>/dev/null | tail -5 || echo "(no trajectory files yet)"
