#!/bin/bash
#SBATCH --job-name=gp-analyze
#SBATCH --account=p_neurasearch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gp/%x-%j.out
# Barnard. Report + verdict -> $OUT/eval/{R,O}/report.txt, verdict.json. Needs gp-rollouts and gp-score.
source slurm/gp/common.sh cpu
bash benchmarks/gp_functions/gonogo.sh analyze
echo "=== verdict R"; cat "$OUT/eval/R/verdict.json"
echo "=== verdict O"; cat "$OUT/eval/O/verdict.json"
