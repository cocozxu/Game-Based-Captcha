#!/bin/bash
# Harness for the trace-the-tunnel replay experiment.
#
# Mirrors experiments/run.sh but reads prompts from experiments_replay/prompts/
# and the solver from experiments_replay/replay_solver.js, so the original
# visual-agent experiment is untouched.
#
# The experiment name determines:
#   - the prompt file:      experiments_replay/prompts/<NAME>.txt
#   - the output folder:    data/<NAME>/
#   - the server mode:      python server.py --experiment <NAME>
#
# Usage:
#   ./experiments_replay/run.sh replay                    # 1 round
#   ./experiments_replay/run.sh replay --rounds 3         # 3 rounds
#
# Prereq: server must already be running with `--experiment <NAME>`.
#   source .venv/bin/activate && python server.py --experiment replay

set -e
cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <experiment_name> [--rounds N]"
  exit 1
fi

NAME="$1"
shift

ROUNDS=1

while [[ $# -gt 0 ]]; do
  case $1 in
    --rounds)
      ROUNDS="$2"
      shift 2
      ;;
    *)
      echo "Unknown flag: $1"
      exit 1
      ;;
  esac
done

PROMPT_FILE="experiments_replay/prompts/${NAME}.txt"
SOLVER_FILE="experiments_replay/replay_solver.js"
OUT_DIR="data/${NAME}"
LOG_DIR="data/logs"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Prompt file not found: $PROMPT_FILE"
  exit 1
fi
if [[ ! -f "$SOLVER_FILE" ]]; then
  echo "Solver file not found: $SOLVER_FILE"
  exit 1
fi

SERVER_MODE=$(curl -s --max-time 2 http://localhost:5050/api/mode || echo "")
if [[ -z "$SERVER_MODE" ]]; then
  echo "Server is not responding at http://localhost:5050."
  echo "Start it first with:"
  echo "  source .venv/bin/activate && python server.py --experiment $NAME"
  exit 1
fi

ACTUAL=$(echo "$SERVER_MODE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('experiment'))")
if [[ "$ACTUAL" != "$NAME" ]]; then
  echo "Server is running in mode '$ACTUAL' but this experiment is '$NAME'."
  echo "Restart the server with: python server.py --experiment $NAME"
  exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"

PROMPT_BODY=$(cat "$PROMPT_FILE")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="$OUT_DIR/manifest_${TIMESTAMP}.json"

EXP_NAME="$NAME" \
EXP_TS="$TIMESTAMP" \
EXP_ROUNDS="$ROUNDS" \
EXP_PROMPT_FILE="$PROMPT_FILE" \
EXP_PROMPT_BODY="$PROMPT_BODY" \
EXP_SOLVER_FILE="$SOLVER_FILE" \
python3 - > "$MANIFEST" <<'PY'
import hashlib, json, os, subprocess
def sha(p):
    with open(p,'rb') as f: return hashlib.sha256(f.read()).hexdigest()[:12]
prompt_file = os.environ["EXP_PROMPT_FILE"]
solver_file = os.environ["EXP_SOLVER_FILE"]
template = open(prompt_file).read()
assembled = os.environ["EXP_PROMPT_BODY"]
try:
    git_sha = subprocess.check_output(["git","rev-parse","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    git_sha = None
print(json.dumps({
    "experiment": os.environ["EXP_NAME"],
    "timestamp": os.environ["EXP_TS"],
    "rounds": int(os.environ["EXP_ROUNDS"]),
    "prompt_file": prompt_file,
    "prompt_sha": hashlib.sha256(template.encode()).hexdigest()[:12],
    "prompt_assembled": assembled,
    "solver_file": solver_file,
    "solver_sha": sha(solver_file),
    "git_sha": git_sha,
}, indent=2))
PY

echo "=== Experiment: $NAME ==="
echo "Rounds:   $ROUNDS"
echo "Prompt:   $PROMPT_FILE"
echo "Solver:   $SOLVER_FILE"
echo "Manifest: $MANIFEST"
echo "Output:   $OUT_DIR/"
echo ""

ALLOWED_TOOLS="mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_evaluate,mcp__plugin_playwright_playwright__browser_run_code_unsafe,mcp__plugin_playwright_playwright__browser_click,mcp__plugin_playwright_playwright__browser_snapshot,Read,Bash"

BEFORE_COUNT=$(ls "$OUT_DIR"/*.json 2>/dev/null | grep -v manifest | wc -l | tr -d ' ')

for round in $(seq 1 "$ROUNDS"); do
  LOGFILE="$LOG_DIR/${NAME}_round${round}_${TIMESTAMP}.txt"
  echo "=== Round $round of $ROUNDS — log: $LOGFILE ==="

  claude -p "$PROMPT_BODY" \
    --verbose \
    --allowedTools "$ALLOWED_TOOLS" \
    2>&1 | tee "$LOGFILE"

  echo ""
  echo "=== Round $round done ==="
  echo ""
done

AFTER_COUNT=$(ls "$OUT_DIR"/*.json 2>/dev/null | grep -v manifest | wc -l | tr -d ' ')
NEW_COUNT=$((AFTER_COUNT - BEFORE_COUNT))
echo "=== Done. New trajectories in $OUT_DIR/: $NEW_COUNT ==="

HUMAN_LEAKS=$(find data/human -type f -newer "$MANIFEST" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$HUMAN_LEAKS" -gt 0 ]]; then
  echo "WARNING: $HUMAN_LEAKS files appeared in data/human/ during this run — investigate."
  find data/human -type f -newer "$MANIFEST"
fi
