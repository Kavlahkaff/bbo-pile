#!/bin/bash
# Barnard login node, repo root, after the rollouts: one report per variant and model seed, and compare.txt, in $WORK/reports.
set -euo pipefail
[[ -f slurm/analyze.sbatch ]] || { echo "run from the repo root" >&2; exit 1; }
source slurm/variants.sh
mkdir -p logs
for v in $VARIANTS; do
  variant "$v"
  for s in $(seeds_of "$v"); do
    export MODEL_SEED=$s
    echo "$v seed $s: $(sbatch --parsable --export=ALL --job-name="sel-analyze-$v-s$s" slurm/analyze.sbatch)"
  done
done
unset MODEL_SEED
echo "compare: $(sbatch --parsable --export=ALL slurm/compare.sbatch)"
