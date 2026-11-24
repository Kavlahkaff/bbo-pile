#!/bin/bash
# Barnard login node, repo root, after the rollouts: one report per variant in $WORK/reports.
set -euo pipefail
[[ -f slurm/analyze.sbatch ]] || { echo "run from the repo root" >&2; exit 1; }
source slurm/variants.sh
mkdir -p logs
for v in $VARIANTS; do
  variant "$v"
  echo "$v: $(sbatch --parsable --export=ALL --job-name="sel-analyze-$v" slurm/analyze.sbatch)"
done
