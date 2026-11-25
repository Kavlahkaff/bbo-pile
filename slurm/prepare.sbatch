#!/bin/bash
#SBATCH --job-name=sel-prepare
#SBATCH --account=p_neurasearch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# Barnard. Export -> tokenizer + checks -> litdata chunks, all under $WORK.
source slurm/common.sh cpu
EXP=$WORK/export/$NAME
python generate_training_data/compile_traces.py --corpus "$CORPUS" --output_path "$EXP" "${MARKER_ARG[@]}" --workers "$SLURM_CPUS_PER_TASK"
if [[ ! -f $TOK/tokenizer.model ]]; then
  # vocabulary = symbols + characters
  head -n 50000 "$EXP/train.txt" > "$WORK/tok_sample_$NAME.txt"
  python generate_training_data/train_tokenizer.py --input_files "$WORK/tok_sample_$NAME.txt" --output_path "$TOK" \
    --vocab_size "$(cat "$VOCAB_FILE")" --input_sentence_size 50000 \
    --extra_symbols "$SYMBOLS"
fi
python generate_training_data/check_tokenizer.py --tokenizer_dir "$TOK" \
  --symbols "$SYMBOLS" --exports "$EXP"
SCRATCH=$WORK/tmp/$SLURM_JOB_ID                     # node-local /tmp is too small
mkdir -p "$SCRATCH"/{tmp,chunks,data}
trap 'rm -rf "$SCRATCH"' EXIT
export TMPDIR=$SCRATCH/tmp DATA_OPTIMIZER_CACHE_FOLDER=$SCRATCH/chunks DATA_OPTIMIZER_DATA_CACHE_FOLDER=$SCRATCH/data
python generate_training_data/preprocess_data.py --input_path "$EXP" --output_path "$DATA" --tokenizer_dir "$TOK" \
  --block_size 1025 --num_workers 4
for s in train valid; do [[ -f $DATA/$s/index.json ]] || { echo "[FATAL] no $DATA/$s/index.json" >&2; exit 1; }; done
