#!/bin/bash
#SBATCH --job-name=sel-compare
#SBATCH --account=p_neurasearch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Variants against base over the model seeds -> $WORK/reports/compare.txt
source slurm/common.sh cpu
source slurm/variants.sh
mkdir -p "$WORK/reports"
ARGS=()
for v in $VARIANTS; do
  for s in $(seeds_of "$v"); do
    run=$(variant "$v"; export MODEL_SEED=$s; source slurm/env.sh; echo "${NAME}_${MODEL}_${TOKENS}$([[ $s == 0 ]] || echo "_s$s")")
    ARGS+=(--run "$v=$WORK/rollouts/$run")
  done
done
python -m evaluation.compare "${ARGS[@]}" --output "$WORK/reports/compare.txt"
