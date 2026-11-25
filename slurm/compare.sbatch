#!/bin/bash
#SBATCH --job-name=sel-compare
#SBATCH --account=p_neurasearch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Runs against $BASE -> $OUT; ARGS and BASE come from submit_analyze.sh
source slurm/common.sh cpu
mkdir -p "$REPORTS"
python -m evaluation.compare --base "$BASE" $ARGS --output "$OUT"
