#!/bin/bash
#SBATCH --job-name=sel-pilot
#SBATCH --account=p_neurasearch
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Barnard. Contrast pilot -> $PILOT. Resubmitting resumes.
source slurm/common.sh cpu
python -m datagen.generate --out "$PILOT" --pilot "$PILOT_FUNCS" --task "$SLURM_ARRAY_TASK_ID" \
  --n_tasks "${N_TASKS:?submit with slurm/submit_pilot.sh}" --workers "$SLURM_CPUS_PER_TASK"
