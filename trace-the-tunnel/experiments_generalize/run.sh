#!/usr/bin/env bash
# Orchestrator for experiments_generalize.
#
# Per (split, model) cell, this script expects:
#   - solvers/<split_id>/<model>.py  exists (you've pasted the agent output)
#   - a server is running with --experiment gen_<model>_<split_id> --expose-debug
#
# Because the server runs in single-experiment mode, you can only do one
# cell per server. This script therefore PROMPTS you to start the right
# server, then runs the dispatch + eval.
#
# Typical workflow:
#   1. python init_splits.py                      # writes splits.yaml
#   2. python render_prompt.py                    # writes prompts/rendered/*.md
#   3. paste each prompt into Opus / Sonnet / Haiku,
#      save returned code to solvers/<split_id>/<model>.py
#   4. for each (split, model):
#        a. in another terminal: python server.py --experiment gen_<model>_<split_id> --expose-debug
#        b. python run_solver.py --split <split_id> --model <model>
#   5. python eval_generalize.py                  # writes results/

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"
source .venv/bin/activate

SPLITS_YAML="$HERE/splits.yaml"
if [[ ! -f "$SPLITS_YAML" ]]; then
    echo "ERROR: $SPLITS_YAML not found. Run: python experiments_generalize/init_splits.py"
    exit 1
fi

# Discover (split, model) cells with a solver in place.
mapfile -t CELLS < <(python - <<'PY'
import os, yaml
here = os.path.dirname(os.path.abspath("experiments_generalize/splits.yaml"))
cfg = yaml.safe_load(open("experiments_generalize/splits.yaml"))
for split in cfg["splits"]:
    for model in cfg["generation"]["models"]:
        p = f"experiments_generalize/solvers/{split['id']}/{model}.py"
        if os.path.exists(p):
            print(f"{split['id']} {model}")
PY
)

if [[ ${#CELLS[@]} -eq 0 ]]; then
    echo "No solvers found under experiments_generalize/solvers/<split>/<model>.py"
    echo "Render prompts (python experiments_generalize/render_prompt.py),"
    echo "paste agent outputs into solvers/<split>/<model>.py, then re-run."
    exit 1
fi

echo "Found ${#CELLS[@]} (split, model) cells with solvers:"
for cell in "${CELLS[@]}"; do echo "  $cell"; done
echo

for cell in "${CELLS[@]}"; do
    split_id="${cell%% *}"
    model="${cell##* }"
    experiment="gen_${model}_${split_id}"
    echo "============================================================"
    echo "Cell: split=$split_id model=$model"
    echo "Required server: python server.py --experiment $experiment --expose-debug"
    echo "============================================================"
    read -r -p "Is the server running with --experiment $experiment? [y/N/skip] " answer
    case "$answer" in
        y|Y)
            python experiments_generalize/run_solver.py --split "$split_id" --model "$model"
            ;;
        skip|s|S)
            echo "skipping cell"
            ;;
        *)
            echo "aborting; start the server and re-run"
            exit 1
            ;;
    esac
done

echo
echo "All requested cells run. Evaluating ..."
python experiments_generalize/eval_generalize.py
echo
echo "Done. See experiments_generalize/results/summary.csv"
