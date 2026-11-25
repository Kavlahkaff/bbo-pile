bash <<'OUTER'
set -euo pipefail
mkdir -p logs
source slurm/variants.sh
for T in 0 0.5 0.7; do
  for v in noUNK log; do
    variant "$v"
    export TEMP=$T MODEL_SEED=0 TAGS="GP NM BEST"
    sbatch --export=ALL --job-name="sel-T$T-$v" <<'EOF'
#!/bin/bash
#SBATCH --account=p_neurasearch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
source slurm/common.sh gpu
python -m evaluation.rollout --checkpoint_dir "$WORK/checkpoints/selection_v1/$RUN/final" --y_norm "$Y_NORM" --tags $TAGS --temperature "$TEMP" --shard "$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT" --output "$WORK/rollouts_T$TEMP/${RUN}_$SLURM_ARRAY_TASK_ID.jsonl"
EOF
  done
done
OUTER
