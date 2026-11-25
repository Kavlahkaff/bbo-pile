#!/bin/bash
#SBATCH --job-name=sel-match-teachers
#SBATCH --account=p_neurasearch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Barnard. Replicated teacher runs and next-point draws -> $WORK/match/teachers.jsonl. Resubmitting resumes.
source slurm/common.sh cpu
python -m evaluation.match teachers --workers "$SLURM_CPUS_PER_TASK" --output "$WORK/match/teachers.jsonl"
