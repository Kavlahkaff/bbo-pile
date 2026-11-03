#!/bin/bash
#SBATCH --job-name=gp-rollouts
#SBATCH --account=p_neurasearch
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gp/%x-%A_%a.out
# Capella. Student rollouts of models R and O on test_core x 5 seeds, shard <task>/<number of tasks>
# -> $OUT/rollouts/{R,O}_shard_*.jsonl. Resubmitting skips finished runs. Needs gp-train.
source slurm/gp/common.sh gpu
bash benchmarks/gp_functions/gonogo.sh rollouts
