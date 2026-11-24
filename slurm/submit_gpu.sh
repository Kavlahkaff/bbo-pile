#!/bin/bash
# Capella login node, repo root, after prepare: train -> rollouts for every variant. Then on Barnard: bash slurm/submit_analyze.sh
set -euo pipefail
[[ -f slurm/train.sbatch ]] || { echo "run from the repo root" >&2; exit 1; }
mkdir -p logs
source slurm/variants.sh
for v in $VARIANTS; do
  variant "$v"
  t=$(sbatch --parsable --export=ALL --job-name="sel-train-$v" slurm/train.sbatch); t=${t%%;*}
  r=$(sbatch --parsable --export=ALL --job-name="sel-rollouts-$v" --kill-on-invalid-dep=yes --dependency=afterok:$t slurm/rollouts.sbatch); r=${r%%;*}
  echo "$v: train=$t rollouts=$r"
done
