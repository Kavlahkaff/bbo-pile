#!/bin/bash
#SBATCH --job-name=sel-analyze
#SBATCH --account=p_neurasearch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Report -> $WORK/reports/$RUN.txt
source slurm/common.sh cpu
mkdir -p "$WORK/reports"
python -m evaluation.analyze --corpus "$CORPUS" --rollouts "$WORK/rollouts/${RUN}_*.jsonl" --output "$WORK/reports/$RUN.txt"
