#!/bin/bash
# Barnard login node, repo root: generate -> check -> prepare all variants (each only if the previous one succeeded).
set -euo pipefail
[[ -f slurm/generate.sbatch ]] || { echo "run from the repo root" >&2; exit 1; }
mkdir -p logs
export N_TASKS=39                      # odd, so every task holds both families
g=$(sbatch --parsable --export=ALL --array=0-$((N_TASKS - 1)) slurm/generate.sbatch); g=${g%%;*}
c=$(sbatch --parsable --export=ALL --kill-on-invalid-dep=yes --dependency=afterok:$g slurm/check.sbatch); c=${c%%;*}
echo "generate=$g check=$c"
AFTER=$c bash slurm/submit_prepare.sh
echo "then on Capella: bash slurm/submit_gpu.sh"
