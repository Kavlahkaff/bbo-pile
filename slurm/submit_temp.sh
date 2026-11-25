#!/bin/bash
# Temperature sweep of trained runs; from the repo root. TEMPS="0.5 0.7 0.85", RUNS="..." as in submit_gpu.sh.
#   bash slurm/submit_temp.sh rollouts   Capella -> $WORK/rollouts_T<t> (resubmitting resumes)
# then on Barnard, per temperature: TEMP=0.7 bash slurm/submit_analyze.sh
set -euo pipefail
[[ -f slurm/rollouts.sbatch ]] || { echo "run from the repo root" >&2; exit 1; }
[[ ${1:-} == rollouts ]] || { echo "usage: bash slurm/submit_temp.sh rollouts" >&2; exit 1; }
source slurm/variants.sh
mkdir -p logs
export TAGS=${TAGS:-"GP NM BEST"}                # random search does not depend on the question
for T in ${TEMPS:-0.5 0.7 0.85}; do
  while read -r -a r; do
    export TEMP=$T
    use "${r[@]}"
    if [[ ! -d $CKPT/final ]]; then echo "skipped $RUN: not trained"; continue; fi
    echo "T=$T $RUN: $(sbatch --parsable --export=ALL --job-name="sel-T$T-$RUN" slurm/rollouts.sbatch </dev/null)"
  done < <(runs)
done
