#!/bin/bash
# Capella login node, repo root, after prepare: train -> rollouts. Then on Barnard: sbatch slurm/analyze.sbatch
set -euo pipefail
[[ -f slurm/train.sbatch ]] || { echo "run from the repo root" >&2; exit 1; }
mkdir -p logs
t=$(sbatch --parsable --export=ALL slurm/train.sbatch); t=${t%%;*}
r=$(sbatch --parsable --export=ALL --kill-on-invalid-dep=yes --dependency=afterok:$t slurm/rollouts.sbatch); r=${r%%;*}
echo "train=$t rollouts=$r"
