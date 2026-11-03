#!/bin/bash
# Run on a Barnard login node from the repo root: bash slurm/gp/submit_barnard.sh
# Submits rank, export, tokenizer, then preprocess once export and tokenizer succeeded.
set -euo pipefail
[[ -f benchmarks/gp_functions/gonogo.sh ]] || { echo "run from the repo root" >&2; exit 1; }
mkdir -p logs/gp
rank=$(sbatch --parsable slurm/gp/rank.sbatch)
exp=$(sbatch --parsable slurm/gp/export.sbatch)
tok=$(sbatch --parsable slurm/gp/tokenizer.sbatch)
pre=$(sbatch --parsable --dependency=afterok:$exp:$tok slurm/gp/preprocess.sbatch)
echo "rank=$rank export=$exp tokenizer=$tok preprocess=$pre"
echo "when preprocess is done: on a Capella login node run  bash slurm/gp/submit_capella.sh"
