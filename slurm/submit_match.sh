#!/bin/bash
# Sampling-distribution match of student and teachers (evaluation/match.py); from the repo root.
#   bash slurm/submit_match.sh teachers   Barnard, once
#   bash slurm/submit_match.sh student    Capella, when teachers is done: every variant and model seed
#   bash slurm/submit_match.sh report     Barnard, when student is done: $WORK/reports/match_<variant>.txt
# Variants: base and noUNK (both have BEST), VARIANTS="base log" for others; SEEDS="0" for one model seed.
set -euo pipefail
[[ -f slurm/match_student.sbatch ]] || { echo "run from the repo root" >&2; exit 1; }
unset NAME TAGS MODEL_SEED
VARIANTS=${VARIANTS:-"base noUNK"}
source slurm/variants.sh
mkdir -p logs
run_of() { (export MODEL_SEED=$1; REPO=$PWD; source slurm/env.sh; echo "${NAME}_${MODEL}_${TOKENS}$([[ $1 == 0 ]] || echo "_s$1")"); }
case "${1:-}" in
teachers) echo "teachers: $(sbatch --parsable --export=ALL slurm/match_teachers.sbatch)" ;;
student)
  for v in $VARIANTS; do
    variant "$v"
    [[ $DATASET != ALL ]] || echo "note: $v has no BEST tag" >&2
    for s in $(seeds_of "$v"); do
      export MODEL_SEED=$s
      echo "$v seed $s: $(sbatch --parsable --export=ALL --job-name="sel-match-$v-s$s" slurm/match_student.sbatch)"
    done
  done ;;
report)
  for v in $VARIANTS; do
    variant "$v"
    export V=$v RUNS=""
    for s in $(seeds_of "$v"); do RUNS+="$(run_of "$s") "; done
    echo "$v: $(sbatch --parsable --export=ALL --job-name="sel-match-report-$v" slurm/match_report.sbatch)"
  done ;;
*) echo "usage: bash slurm/submit_match.sh teachers|student|report" >&2; exit 1 ;;
esac
