#!/bin/bash
# End-to-end harness for the RL agent experiment.
#
# Usage:
#   ./experiments_rl/run_rl.sh train              # BC + PPO training only
#   ./experiments_rl/run_rl.sh collect            # collect data (server must be running)
#   ./experiments_rl/run_rl.sh all                # train then collect
#
# Options:
#   --bc-epochs N       BC training epochs (default: 150)
#   --ppo-iters N       PPO iterations (default: 200)
#   --rounds N          Collection rounds over all tunnels (default: 3)
#   --policy FILE       Policy checkpoint to use for collection (default: ppo_policy_best.pt)
#   --headless          Run browser headlessly during collection
#
# The server must already be running in experiment mode for 'collect':
#   source .venv/bin/activate
#   python server.py --experiment rl_agent
#
# Example full run:
#   Terminal A: source .venv/bin/activate && python server.py --experiment rl_agent
#   Terminal B: ./experiments_rl/run_rl.sh all --ppo-iters 300 --rounds 5

set -e
cd "$(dirname "$0")/.."

# --- Parse command ---
CMD="${1:-all}"
shift || true

BC_EPOCHS=150
PPO_ITERS=200
ROUNDS=3
POLICY="ppo_policy_best.pt"
HEADLESS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bc-epochs)  BC_EPOCHS="$2";  shift 2 ;;
    --ppo-iters)  PPO_ITERS="$2";  shift 2 ;;
    --rounds)     ROUNDS="$2";     shift 2 ;;
    --policy)     POLICY="$2";     shift 2 ;;
    --headless)   HEADLESS="--headless"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# --- Activate venv ---
if [[ ! -d ".venv" ]]; then
  echo "ERROR: .venv not found. Run ./setup.sh first."
  exit 1
fi
source .venv/bin/activate

EXP_NAME="rl_agent"
MODEL_DIR="experiments_rl/models"
mkdir -p "$MODEL_DIR"

# ---------------------------------------------------------------------------
run_train() {
  echo ""
  echo "=== Phase 1: Behavioral Cloning ==="
  echo "  Epochs: $BC_EPOCHS"
  python experiments_rl/bc_train.py --epochs "$BC_EPOCHS"

  echo ""
  echo "=== Phase 2: PPO Fine-tuning ==="
  echo "  Iterations: $PPO_ITERS"
  python experiments_rl/ppo_train.py --iters "$PPO_ITERS"
}

run_collect() {
  echo ""
  echo "=== Data Collection ==="

  # Verify server
  SERVER_MODE=$(curl -s --max-time 2 http://localhost:5050/api/mode || echo "")
  if [[ -z "$SERVER_MODE" ]]; then
    echo "ERROR: Server not responding at http://localhost:5050"
    echo "Start it with: source .venv/bin/activate && python server.py --experiment $EXP_NAME"
    exit 1
  fi
  ACTUAL=$(echo "$SERVER_MODE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('experiment',''))")
  if [[ "$ACTUAL" != "$EXP_NAME" ]]; then
    echo "ERROR: Server in mode '$ACTUAL', expected '$EXP_NAME'"
    exit 1
  fi

  POLICY_PATH="$MODEL_DIR/$POLICY"
  if [[ ! -f "$POLICY_PATH" ]]; then
    echo "ERROR: Policy not found at $POLICY_PATH"
    echo "Run training first: ./experiments_rl/run_rl.sh train"
    exit 1
  fi

  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  LOGFILE="data/logs/${EXP_NAME}_${TIMESTAMP}.txt"
  mkdir -p data/logs

  echo "  Policy:     $POLICY_PATH"
  echo "  Rounds:     $ROUNDS"
  echo "  Output:     data/$EXP_NAME/"
  echo "  Log:        $LOGFILE"

  BEFORE=$(ls "data/$EXP_NAME/"*.json 2>/dev/null | grep -v manifest | wc -l | tr -d ' ')

  python experiments_rl/run_agent.py \
    --policy "$POLICY" \
    --rounds "$ROUNDS" \
    --experiment "$EXP_NAME" \
    $HEADLESS \
    2>&1 | tee "$LOGFILE"

  AFTER=$(ls "data/$EXP_NAME/"*.json 2>/dev/null | grep -v manifest | wc -l | tr -d ' ')
  echo ""
  echo "=== Collected $((AFTER - BEFORE)) new trajectories in data/$EXP_NAME/ ==="

  # Sanity: no human data leakage
  LEAKS=$(find data/human -newer "$LOGFILE" -type f 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$LEAKS" -gt 0 ]]; then
    echo "WARNING: $LEAKS unexpected files in data/human/ — investigate!"
  fi
}

# ---------------------------------------------------------------------------
case "$CMD" in
  train)   run_train ;;
  collect) run_collect ;;
  all)     run_train && run_collect ;;
  *)
    echo "Usage: $0 {train|collect|all} [options]"
    exit 1
    ;;
esac
