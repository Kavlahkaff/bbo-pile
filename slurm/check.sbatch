#!/bin/bash
#SBATCH --job-name=sel-check
#SBATCH --account=p_neurasearch
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Barnard. Completeness and optimizer contrast of the corpus; the later jobs run only if this passes.
source slurm/common.sh cpu
python -m datagen.check_corpus "$CORPUS" --train "$TRAIN_FUNCS"
