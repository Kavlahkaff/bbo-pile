#!/bin/bash
#SBATCH --job-name=gp-export
#SBATCH --account=p_neurasearch
#SBATCH --array=0-1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gp/%x-%A_%a.out
# Barnard. Exports R (task 0) and O (task 1), shuffled -> $OUT/export/{R,O}.
ARRAY_OVER_DATASETS=1
source slurm/gp/common.sh cpu
bash benchmarks/gp_functions/gonogo.sh export
