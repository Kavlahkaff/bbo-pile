#!/bin/bash
#SBATCH --job-name=gp-rank
#SBATCH --account=p_neurasearch
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gp/%x-%j.out
# Barnard. Step 0 teacher ranking -> $OUT/ranks (teacher_ranks.parquet, report, B_np family mapping).
source slurm/gp/common.sh cpu
bash benchmarks/gp_functions/gonogo.sh rank
