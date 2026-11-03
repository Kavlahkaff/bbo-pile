#!/bin/bash
# Run on a Capella login node from the repo root, after the Barnard preprocessing finished:
#   bash slurm/gp/submit_capella.sh
# Submits train (R, O), then rollouts and scoring once training succeeded.
set -euo pipefail
[[ -f benchmarks/gp_functions/gonogo.sh ]] || { echo "run from the repo root" >&2; exit 1; }
mkdir -p logs/gp
train=$(sbatch --parsable slurm/gp/train.sbatch)
roll=$(sbatch --parsable --dependency=afterok:$train slurm/gp/rollouts.sbatch)
score=$(sbatch --parsable --dependency=afterok:$train slurm/gp/score.sbatch)
echo "train=$train rollouts=$roll score=$score"
echo "when rollouts and score are done: on a Barnard login node run  mkdir -p logs/gp && sbatch slurm/gp/analyze.sbatch"
