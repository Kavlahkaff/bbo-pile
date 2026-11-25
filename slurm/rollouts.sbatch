#!/bin/bash
#SBATCH --job-name=sel-rollouts
#SBATCH --account=p_neurasearch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Capella. Closed-loop rollouts on the test functions. Resubmitting resumes.
source slurm/common.sh gpu
python -m evaluation.rollout --checkpoint_dir "$CKPT/final" "${MARKER_ARG[@]}" --tags $TAGS --temperature "$TEMP" \
  --shard "$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT" --output "$ROLLOUTS/${RUN}_$SLURM_ARRAY_TASK_ID.jsonl"
