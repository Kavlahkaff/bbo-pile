#!/bin/bash
#SBATCH --job-name=sel-match-student
#SBATCH --account=p_neurasearch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Capella. Replicated student runs and next-point draws -> $WORK/match/${RUN}_<shard>.jsonl. Resubmitting resumes.
source slurm/common.sh gpu
TAGS=$(for t in $TAGS; do [[ $t == RS ]] || echo -n "$t "; done)   # the RS tag is not evaluated here
python -m evaluation.match student --checkpoint_dir "$WORK/checkpoints/selection_v1/$RUN/final" --y_norm "$Y_NORM" --tags $TAGS \
  --teachers "$WORK/match/teachers.jsonl" --shard "$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT" \
  --output "$WORK/match/${RUN}_$SLURM_ARRAY_TASK_ID.jsonl"
