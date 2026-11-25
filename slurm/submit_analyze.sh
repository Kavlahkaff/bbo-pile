#!/bin/bash
# Barnard login node, repo root, after the rollouts: a report per run, and compare_<time>.txt, in $WORK/reports (reports_T<t> with TEMP=<t>).
# Reports that exist are kept; BASE=<label> sets the baseline of the comparison.
set -euo pipefail
[[ -f slurm/analyze.sbatch ]] || { echo "run from the repo root" >&2; exit 1; }
source slurm/variants.sh
mkdir -p logs
export BASE=${BASE:-log_qwen3_5M_2.4B_lr5e-3} ARGS=""
while read -r -a r; do
  use "${r[@]}"
  if [[ ! -e $ROLLOUTS/${RUN}_0.jsonl ]]; then echo "skipped $RUN: no rollouts"; continue; fi
  ARGS+=" --run $(label "${r[@]}")=$ROLLOUTS/$RUN"
  if [[ -e $REPORTS/$RUN.txt ]]; then echo "kept $REPORTS/$RUN.txt"; continue; fi
  echo "$RUN: $(sbatch --parsable --export=ALL --job-name="sel-analyze-$RUN" slurm/analyze.sbatch </dev/null)"
done < <(runs)
[[ -n $ARGS ]] || { echo "no rollouts found" >&2; exit 1; }
export OUT=$REPORTS/compare_$(date +%Y%m%d_%H%M%S).txt
echo "compare -> $OUT: $(sbatch --parsable --export=ALL slurm/compare.sbatch)"
