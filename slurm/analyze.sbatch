#!/bin/bash
#SBATCH --job-name=sel-analyze
#SBATCH --account=p_neurasearch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Report -> $REPORTS/$RUN.txt
source slurm/common.sh cpu
mkdir -p "$REPORTS"
python -m evaluation.analyze --corpus "$CORPUS" --rollouts "$ROLLOUTS/${RUN}_[0-9]*.jsonl" --output "$REPORTS/$RUN.txt"
