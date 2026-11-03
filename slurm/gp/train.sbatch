#!/bin/bash
#SBATCH --job-name=gp-train
#SBATCH --account=p_neurasearch
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gp/%x-%A_%a.out
# Capella. Generates the config and trains the 5M model on R (task 0) and O (task 1)
# -> $BASE/checkpoints/gp_v1/gonogo_{R,O}_qwen3_5M_token_${TOKENS}_lr_${LR}_*/final
ARRAY_OVER_DATASETS=1
source slurm/gp/common.sh gpu
bash benchmarks/gp_functions/gonogo.sh train
