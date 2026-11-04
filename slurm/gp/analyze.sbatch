#!/bin/bash
#SBATCH --job-name=gp-analyze
#SBATCH --account=p_neurasearch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gp/%x-%j.out
# Barnard. Per label report + verdict -> $OUT/eval/<label>; arm comparison -> $OUT/compare/<REPORT_NAME>;
# plots -> $OUT/plots/<REPORT_NAME>. Needs gp-rollouts (and gp-score for the tag-likelihood part).
source slurm/gp/common.sh cpu
bash benchmarks/gp_functions/gonogo.sh analyze
for e in ${LABELS:-${DATASETS:-R O}}; do
  l=${e%%@*}; [[ -f "$OUT/eval/$l/verdict.json" ]] && { echo "=== verdict $l"; cat "$OUT/eval/$l/verdict.json"; echo; }
done
