#!/bin/bash
# Barnard login node, repo root: contrast pilot, then its report (only if every task succeeded).
set -euo pipefail
[[ -f slurm/pilot.sbatch ]] || { echo "run from the repo root" >&2; exit 1; }
mkdir -p logs
export N_TASKS=19                      # coprime to 5 families and 4 kernels
p=$(sbatch --parsable --export=ALL --array=0-$((N_TASKS - 1)) slurm/pilot.sbatch); p=${p%%;*}
r=$(sbatch --parsable --export=ALL --kill-on-invalid-dep=yes --dependency=afterok:$p slurm/pilot_report.sbatch); r=${r%%;*}
echo "pilot=$p report=$r"
