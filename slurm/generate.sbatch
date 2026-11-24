#!/bin/bash
#SBATCH --job-name=sel-generate
#SBATCH --account=p_neurasearch
#SBATCH --array=0-39
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Barnard. Trajectory corpus -> $CORPUS. Resubmitting resumes.
source slurm/common.sh cpu
python -m datagen.generate --out "$CORPUS" --train "$TRAIN_FUNCS" --task "$SLURM_ARRAY_TASK_ID" \
  --n_tasks "${N_TASKS:?submit with slurm/submit_cpu.sh}" --workers "$SLURM_CPUS_PER_TASK"
