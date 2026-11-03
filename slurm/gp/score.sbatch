#!/bin/bash
#SBATCH --job-name=gp-score
#SBATCH --account=p_neurasearch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gp/%x-%j.out
# Capella. Teacher-forced tag-likelihood scores of model R on test_core -> $OUT/scores_R.parquet. Needs gp-train.
source slurm/gp/common.sh gpu
bash benchmarks/gp_functions/gonogo.sh score
