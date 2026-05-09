#!/bin/bash
# Generic harness for running a trace-the-tunnel agent experiment.
#
# Each experiment is identified by a name, which determines:
#   - the prompt file:      experiments/prompts/<NAME>.txt
#   - the output folder:    data/<NAME>/
#   - the server mode:      python server.py --experiment <NAME>
#
# The server enforces folder routing — the agent literally cannot write
# anywhere else.
#
# Usage:
#   ./experiments/run.sh <NAME>                    # 1 round
#   ./experiments/run.sh <NAME> --rounds 3         # 3 rounds
#   ./experiments/run.sh <NAME> --examples a.png b.png c.png   # attach images
#
# Prereq: the server must already be running with `--experiment <NAME>`.
# Start it in another terminal:
#   source .venv/bin/activate && python server.py --experiment <NAME>

set -e
cd "$(dirname "$0")/.."

# --- Parse args ---
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <experiment_name> [--rounds N] [--examples img1 img2 ...]"
  exit 1
fi

NAME="$1"
shift

ROUNDS=1
EXAMPLE_FILES=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --rounds)
      ROUNDS="$2"
      shift 2
      ;;
    --examples)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        EXAMPLE_FILES+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown flag: $1"
      exit 1
      ;;
  esac
done

PROMPT_FILE="experiments/prompts/${NAME}.txt"
OUT_DIR="data/${NAME}"
LOG_DIR="data/logs"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Prompt file not found: $PROMPT_FILE"
  exit 1
fi

# --- Verify server is up and in the matching experiment mode ---
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

# --- Build prompt body, injecting example image paths if any ---
# `claude -p` has no --image flag; the agent ingests local images via Read.
PROMPT_BODY=$(cat "$PROMPT_FILE")

if [[ ${#EXAMPLE_FILES[@]} -gt 0 ]]; then
  for img in "${EXAMPLE_FILES[@]}"; do
    if [[ ! -f "$img" ]]; then
      echo "Example image not found: $img"
      exit 1
    fi
  done
  EXAMPLES_BLOCK=$'\n\nBefore you start playing, use the Read tool to view these example screenshots from past sessions:\n'
  for img in "${EXAMPLE_FILES[@]}"; do
    EXAMPLES_BLOCK+="  - $(pwd)/$img"$'\n'
  done
  PROMPT_BODY="${PROMPT_BODY}${EXAMPLES_BLOCK}"
fi

# --- Manifest: capture exactly what was run ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="$OUT_DIR/manifest_${TIMESTAMP}.json"

# Pass the example list and the assembled prompt to Python through env vars.
EXAMPLES_NL=$(printf "%s\n" "${EXAMPLE_FILES[@]}")
EXP_NAME="$NAME" \
EXP_TS="$TIMESTAMP" \
EXP_ROUNDS="$ROUNDS" \
EXP_PROMPT_FILE="$PROMPT_FILE" \
EXP_PROMPT_BODY="$PROMPT_BODY" \
EXP_EXAMPLES="$EXAMPLES_NL" \
python3 - > "$MANIFEST" <<'PY'
import hashlib, json, os, subprocess
def sha(p):
    with open(p,'rb') as f: return hashlib.sha256(f.read()).hexdigest()[:12]
prompt_file = os.environ["EXP_PROMPT_FILE"]
template = open(prompt_file).read()
assembled = os.environ["EXP_PROMPT_BODY"]
examples = [p for p in os.environ.get("EXP_EXAMPLES","").splitlines() if p]
try:
    git_sha = subprocess.check_output(["git","rev-parse","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    git_sha = None
print(json.dumps({
    "experiment": os.environ["EXP_NAME"],
    "timestamp": os.environ["EXP_TS"],
    "rounds": int(os.environ["EXP_ROUNDS"]),
    "prompt_file": prompt_file,
    "prompt_template_sha": hashlib.sha256(template.encode()).hexdigest()[:12],
    "prompt_assembled_sha": hashlib.sha256(assembled.encode()).hexdigest()[:12],
    "prompt_assembled": assembled,
    "examples": [{"path": p, "sha": sha(p)} for p in examples],
    "git_sha": git_sha,
}, indent=2))
PY

echo "=== Experiment: $NAME ==="
echo "Rounds: $ROUNDS"
echo "Prompt: $PROMPT_FILE"
echo "Examples: ${EXAMPLE_FILES[*]:-(none)}"
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

# --- Sanity check: count new trajectory files, confirm none leaked elsewhere ---
AFTER_COUNT=$(ls "$OUT_DIR"/*.json 2>/dev/null | grep -v manifest | wc -l | tr -d ' ')
NEW_COUNT=$((AFTER_COUNT - BEFORE_COUNT))
echo "=== Done. New trajectories in $OUT_DIR/: $NEW_COUNT ==="

# Belt-and-suspenders: warn if anything landed in data/human/ since we started.
HUMAN_LEAKS=$(find data/human -type f -newer "$MANIFEST" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$HUMAN_LEAKS" -gt 0 ]]; then
  echo "WARNING: $HUMAN_LEAKS files appeared in data/human/ during this run — investigate."
  find data/human -type f -newer "$MANIFEST"
fi
