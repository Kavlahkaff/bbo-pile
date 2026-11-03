#!/bin/bash
#SBATCH --job-name=gp-preprocess
#SBATCH --account=p_neurasearch
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gp/%x-%A_%a.out
# Barnard. Tokenizes R (task 0) and O (task 1) into litdata chunks -> $BASE/tokenized_dataset/gp_v1/{R,O}.
# Needs gp-export and gp-tokenizer.
ARRAY_OVER_DATASETS=1
source slurm/gp/common.sh cpu
bash benchmarks/gp_functions/gonogo.sh preprocess
