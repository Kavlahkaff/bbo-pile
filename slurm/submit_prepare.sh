#!/bin/bash
# Barnard login node, repo root: export + tokenize every variant. AFTER=<jobid> waits for that job.
# The first variant makes the shared tokenizer, so the others wait for it.
set -euo pipefail
[[ -f slurm/prepare.sbatch ]] || { echo "run from the repo root" >&2; exit 1; }
source slurm/variants.sh
mkdir -p logs
dep=${AFTER:-}; first=1
for v in $VARIANTS; do
  variant "$v"
  j=$(sbatch --parsable --export=ALL ${dep:+--kill-on-invalid-dep=yes --dependency=afterok:$dep} slurm/prepare.sbatch); j=${j%%;*}
  echo "prepare $v=$j"
  if [[ $first == 1 ]]; then dep=$j; first=0; fi
done
