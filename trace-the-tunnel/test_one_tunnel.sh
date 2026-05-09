#!/bin/bash
# Test a single tunnel run with Claude Code as the agent.
# Prerequisites: server running on port 5050
#
# Usage:
#   ./test_one_tunnel.sh        # plays tunnel 0
#   ./test_one_tunnel.sh 3      # plays tunnel 3

set -e
cd "$(dirname "$0")"

TID=${1:-0}
mkdir -p data/logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="data/logs/test_tunnel${TID}_${TIMESTAMP}.txt"

echo "=== Testing tunnel $TID ==="
echo "Make sure the server is running: source .venv/bin/activate && python server.py"
echo "Log: $LOGFILE"
echo ""

claude -p "You are playing the trace-the-tunnel game to collect agent trajectory data.
The game server is running at http://localhost:5050.
Read the CLAUDE.md file in this directory for reference.

Your task: Play tunnel ID $TID. Save with source='agent'.

1. Navigate to http://localhost:5050
2. Load tunnel $TID
3. Read the tunnel geometry
4. Trace through the tunnel. Try to move the way a real human would.
5. After the game ends (success or failure), re-save with source='agent'.
6. Confirm the save.

Play the tunnel now." --verbose --allowedTools "mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_evaluate,mcp__plugin_playwright_playwright__browser_run_code_unsafe,mcp__plugin_playwright_playwright__browser_click,mcp__plugin_playwright_playwright__browser_snapshot,Read,Bash" 2>&1 | tee "$LOGFILE"

echo ""
echo "=== Done. Log: $LOGFILE ==="
ls -la data/agent/ 2>/dev/null | tail -3 || echo "(no files yet)"
