#!/bin/bash
#SBATCH --job-name=gp-tokenizer
#SBATCH --account=p_neurasearch
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/gp/%x-%j.out
# Barnard. Mixed sample of every tag scheme -> tokenizer gp_v1 in $BASE/tokenizer + checks ($OUT/tokenizer_check.json).
source slurm/gp/common.sh cpu
bash benchmarks/gp_functions/gonogo.sh tokenizer
