#!/bin/bash
# Barnard login node, repo root: export + tokenize. VARIANTS="log_marker" selects; existing data is skipped. AFTER=<jobid> waits for that job.
set -euo pipefail
[[ -f slurm/prepare.sbatch ]] || { echo "run from the repo root" >&2; exit 1; }
source slurm/variants.sh
unset TEMP TAGS
mkdir -p logs
for v in ${VARIANTS:-log log_marker}; do
  use "$v" qwen3_5M 2.4B 5e-3 0
  if [[ -e $DATA || -e $WORK/export/$NAME ]]; then echo "exists, skipped: $NAME"; continue; fi
  if [[ -n ${AFTER:-} ]]; then dep="--kill-on-invalid-dep=yes --dependency=afterok:$AFTER"; else dep=; fi
  echo "prepare $v=$(sbatch --parsable --export=ALL --job-name="sel-prepare-$v" $dep slurm/prepare.sbatch)"
done
