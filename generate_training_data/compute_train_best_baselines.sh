#!/bin/bash
#SBATCH --job-name=train_best_baselines
#SBATCH --output=../logs/train_best_baselines_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Computes the per-task "best of the six baselines" threshold
# (find_best_baselines.py's mean_val) scoped to Approach C's TRAINING tasks
# (all_training_tasks.json), not find_best_baselines.py's default
# (validation_tasks.json, the held-out set). select_self_play_rollouts.py
# needs this threshold to decide whether a self-play rollout "beats the best
# of the six baseline optimizers" on the training task it was generated for.
#
# Run once (or whenever raw_data_bbo_pile / all_training_tasks.json change),
# not once per self-play round -- the threshold is a property of the offline
# baseline logs, independent of the current checkpoint being self-played.

set -euo pipefail
trap 'echo "[FATAL] line $LINENO: command failed: $BASH_COMMAND" >&2' ERR

module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0

cd "$(dirname "$0")"
source ../.venv/bin/activate
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[FATAL] failed to activate virtualenv .venv" >&2
    exit 1
fi

export PYTHONPATH=/data/horse/ws/luth474h-master_thesis/bbo-pile:${PYTHONPATH:-}
export RESULTS_PATH=/data/horse/ws/luth474h-master_thesis/raw_data_bbo_pile

ALL_TRAINING_TASKS_JSON="all_training_tasks.json"
OUT_JSON="train_best_baselines.json"

if [[ ! -f "$ALL_TRAINING_TASKS_JSON" ]]; then
    echo "[FATAL] $ALL_TRAINING_TASKS_JSON not found -- run build_all_training_tasks.py first" >&2
    exit 1
fi

python -u find_best_baselines.py \
    --results_path "$RESULTS_PATH" \
    --all_training_tasks_json "$ALL_TRAINING_TASKS_JSON" \
    --out_json "$OUT_JSON"

echo "[INFO] wrote $OUT_JSON"
