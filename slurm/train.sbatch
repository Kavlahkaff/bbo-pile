#!/bin/bash
#SBATCH --job-name=sel-train
#SBATCH --account=p_neurasearch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Capella. Trains $MODEL on data set $NAME -> $CKPT
source slurm/common.sh gpu
export LITDATA_POSIX_FAST=0 #should prevent dropping the workerwith the least data to avoid data loss
CFG_DIR=configs/generated/$RUN
rm -rf "$CFG_DIR"
python configs/generate_configs.py --base_path "$WORK" --version selection_v1 --dataset "$NAME" --models "$MODEL" --tokens "$TOKENS" \
  --lrs "$LR" --seeds "$MODEL_SEED" --bsz 16 --vocab_size "$(cat "$VOCAB_FILE")" --max_seq_length 1024 \
  --min_tokens_per_param 0 --run_prefix "${NAME}_" --out_dir "$CFG_DIR"
CFG=$(ls "$CFG_DIR"/*.yaml)
sed -i "s#^tokenizer_dir: .*#tokenizer_dir: $TOK#; s#^out_dir: .*#out_dir: $CKPT#" "$CFG"
LOGGER=(); [[ -n "${WANDB_API_KEY:-}" ]] || LOGGER=(--logger_name csv)
python open_optformer/training/pretrain.py "Qwen3-${MODEL#qwen3_}" --config "$CFG" "${LOGGER[@]}"
