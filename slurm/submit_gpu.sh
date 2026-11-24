#!/bin/bash
# Capella login node, repo root, after prepare: train -> rollouts for every variant and model seed. Then on Barnard: bash slurm/submit_analyze.sh
set -euo pipefail
[[ -f slurm/train.sbatch ]] || { echo "run from the repo root" >&2; exit 1; }
mkdir -p logs
source slurm/variants.sh
for v in $VARIANTS; do
  variant "$v"
  for s in $(seeds_of "$v"); do
    export MODEL_SEED=$s
    t=$(sbatch --parsable --export=ALL --job-name="sel-train-$v-s$s" slurm/train.sbatch); t=${t%%;*}
    r=$(sbatch --parsable --export=ALL --job-name="sel-rollouts-$v-s$s" --kill-on-invalid-dep=yes --dependency=afterok:$t slurm/rollouts.sbatch); r=${r%%;*}
    echo "$v seed $s: train=$t rollouts=$r"
  done
done
