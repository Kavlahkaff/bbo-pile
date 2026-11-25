#!/bin/bash
#SBATCH --job-name=sel-match-report
#SBATCH --account=p_neurasearch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Barnard. -> $WORK/reports/match_<variant>.txt and .csv.gz (all distances); $V and $RUNS come from submit_match.sh
source slurm/common.sh cpu
mkdir -p "$WORK/reports"
ARGS=()
for r in $RUNS; do ARGS+=(--run "$V=$WORK/match/$r"); done
python -m evaluation.match report --teachers "$WORK/match/teachers.jsonl" "${ARGS[@]}" --workers "$SLURM_CPUS_PER_TASK" \
  --output "$WORK/reports/match_$V.txt"
