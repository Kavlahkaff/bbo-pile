#!/bin/bash
# Sampling-distribution match of student and teachers (evaluation/match.py); from the repo root.
#   bash slurm/submit_match.sh teachers   Barnard, once (done: $WORK/match/teachers.jsonl)
#   bash slurm/submit_match.sh student    Capella: every trained run (RUNS="..." selects)
#   bash slurm/submit_match.sh report     Barnard, when student is done: $WORK/reports/match_<label>.txt
set -euo pipefail
[[ -f slurm/match_student.sbatch ]] || { echo "run from the repo root" >&2; exit 1; }
source slurm/variants.sh
unset TEMP TAGS
mkdir -p logs
case "${1:-}" in
teachers) echo "teachers: $(sbatch --parsable --export=ALL slurm/match_teachers.sbatch)" ;;
student)
  while read -r -a r; do
    use "${r[@]}"
    if [[ ! -d $CKPT/final ]]; then echo "skipped $RUN: not trained"; continue; fi
    echo "$RUN: $(sbatch --parsable --export=ALL --job-name="sel-match-$RUN" slurm/match_student.sbatch </dev/null)"
  done < <(runs) ;;
report)
  LIST=$(while read -r -a r; do
    use "${r[@]}"
    if [[ -e $WORK/match/${RUN}_0.jsonl ]]; then echo "$(label "${r[@]}") $RUN"; fi
  done < <(runs))
  [[ -n $LIST ]] || { echo "no match results found" >&2; exit 1; }
  for v in $(awk '{print $1}' <<< "$LIST" | sort -u); do
    export V=$v RUNS_OF; RUNS_OF=$(awk -v v="$v" '$1 == v {print $2}' <<< "$LIST" | tr '\n' ' ')
    echo "$v: $(sbatch --parsable --export=ALL --job-name="sel-match-report-$v" slurm/match_report.sbatch </dev/null)"
  done ;;
*) echo "usage: bash slurm/submit_match.sh teachers|student|report" >&2; exit 1 ;;
esac
