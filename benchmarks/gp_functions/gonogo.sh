#!/usr/bin/env bash
# Early go/no-go check (before building the full experiment grid):
#   export O and R (no symmetries), train a 5M model on each, then on test_core check
#   (a) the student prompted as the oracle vs the matched-kernel MLE teacher, floored at 1e-3 (rollouts)
#   (b) whether the tag-likelihood matrix separates oracle / MLE / RS (teacher-forced scores, model R)
#
# On the cluster, submit the SLURM jobs in slurm/gp/ instead of calling this directly (see slurm/gp/README.md).
#
# Usage: bash benchmarks/gp_functions/gonogo.sh <stage>     stages, in order:
#   rank        step 0 teacher ranking                                             -> $OUT/ranks
#   export      exports $DATASETS (default "R O"), shuffled                        -> $OUT/export/<d>
#   tokenizer   small exports of every tag scheme -> mixed sample -> gp_v1 + checks -> $BASE/tokenizer/gp_v1
#   preprocess  tokenize $DATASETS into litdata chunks                             -> $BASE/tokenized_dataset/gp_v1/<d>
#   configs     training configs (5M, $TOKENS, lr $LR, bsz 16)                     -> configs/gp_gonogo/*.yaml
#   train       configs + pretrain for $DATASETS (GPU)                             -> $BASE/checkpoints/gp_v1/gonogo_<d>_*
#   rollouts    student rollouts on test_core x 5 seeds, shard $SHARD (i/n)        -> $OUT/rollouts/<d>_shard_*.jsonl
#   score       teacher-forced tag-likelihood scores of $SCORE_DATASETS (R) on test_core -> $OUT/scores_<d>.parquet
#   analyze     per label: report + verdict.json -> $OUT/eval/<label>; with >= 2 labels the paired arm comparison
#               (first label = reference) -> $OUT/compare/<name>; trajectory plots -> $OUT/plots/<name>
#
# Datasets: R, O, and the O ablations O_none / O_random / O_causal (prefix lines, y normalization; see ARGS below).
# Environment: CORPUS, BASE, OUT (see slurm/gp/env.sh), optionally TOKENS (300M), LR (5e-3), DATASETS ("R O"),
# ROLLOUT_TAGS (space separated templates, default all five), SCORE_DATASETS ("R"), MODEL (qwen3_5M),
# ROLLOUT_TEMPERATURE (1; 0 = greedy), LABELS / REPORT_NAME / PLOT_TAGS for analyze. Rollouts of a model are stored
# under its label: dataset + model size / tokens / temperature where not the default (R, R_10M_1.2B, O_T0.5, ...).
# SHARD (i/n; in a SLURM array: task id / task count), CKPT_R / CKPT_O (default: the gonogo checkpoints in $BASE).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CORPUS=${CORPUS:-/data/horse/ws/luth474h-fmbo/fmbo-data-gen/corpus_student_40k}
BASE=${BASE:-/projects/p_neurasearch/bbo-pile_experiments}
OUT=${OUT:-/data/horse/ws/luth474h-fmbo/gp_gonogo}
VERSION=gp_v1
VOCAB=${VOCAB:-1062}
MODEL=${MODEL:-qwen3_5M}
TOKENS=${TOKENS:-300M}
LR=${LR:-5e-3}
TEMPERATURE=${ROLLOUT_TEMPERATURE:-1}
DATASETS=${DATASETS:-R O}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  SHARD=${SHARD:-$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT}
fi
SHARD=${SHARD:-0/1}
if [[ -n "${ROLLOUT_TAGS:-}" ]]; then read -r -a TAGS <<< "$ROLLOUT_TAGS"
else TAGS=('GP-{K}-ORACLE' 'GP-{K}-MLE' 'RS' 'UNK' 'RS-baseline'); fi
EXPORT="python generate_training_data/compile_gp_data.py --corpus $CORPUS --workers $WORKERS"
# reference R: kernel_mode tags, all teachers, tag dropout 0.2, fixed prefixes, no symmetries (S1); O = oracle only.
# O_* = prefix / y-normalization ablation of O (each changes one thing vs O; O_causal has no prefixes, which are
# redundant under causal normalization)
declare -A ARGS=(
  [R]="--tag_scheme kernel_mode --teachers all --tag_dropout 0.2 --prefix_mode fixed"
  [O]="--tag_scheme kernel_mode --teachers oracle --tag_dropout 0.2 --prefix_mode fixed"
  [O_none]="--tag_scheme kernel_mode --teachers oracle --tag_dropout 0.2 --prefix_mode none"
  [O_random]="--tag_scheme kernel_mode --teachers oracle --tag_dropout 0.2 --prefix_mode random --n_prefixes 2"
  [O_causal]="--tag_scheme kernel_mode --teachers oracle --tag_dropout 0.2 --prefix_mode none --y_norm causal"
)
# y normalization a model was trained with (rollouts and scoring must prompt it the same way)
ynorm_of() { [[ "${ARGS[$1]}" == *"--y_norm causal"* ]] && echo causal || echo global; }
RUN_PREFIX=gonogo
CONFIG_DIR=configs/gp_gonogo

shuffle() { for s in train valid; do shuf "$1/$s.txt" -o "$1/$s.txt"; done; }
config_of() { ls $CONFIG_DIR/${RUN_PREFIX}_$1_${MODEL}_token_${TOKENS}_lr_${LR}_bsz_16_*_seed_0.yaml 2>/dev/null | head -1; }
# label of a model's rollouts / reports: the dataset, plus model size / tokens / temperature where not the default
# (R, R_1.2B, R_10M_1.2B, R_T0.5, O_greedy, ...)
label_of() {
  local l=$1
  [[ $MODEL != qwen3_5M ]] && l+="_${MODEL#qwen3_}"
  [[ $TOKENS != 300M ]] && l+="_$TOKENS"
  case $TEMPERATURE in 1|1.0) ;; 0|0.0) l+="_greedy" ;; *) l+="_T$TEMPERATURE" ;; esac
  echo "$l"
}
ckpt_of() {
  local v=CKPT_$1
  if [[ -n "${!v:-}" ]]; then echo "${!v}"; return; fi
  local c; c=$(ls -d $BASE/checkpoints/$VERSION/${RUN_PREFIX}_$1_${MODEL}_token_${TOKENS}_lr_${LR}_bsz_16_*_seed_0/final 2>/dev/null | head -1)
  [[ -n "$c" ]] || { echo "[FATAL] no checkpoint for $1 under $BASE/checkpoints/$VERSION (set CKPT_$1)" >&2; exit 1; }
  echo "$c"
}
make_configs() {
  for d in $DATASETS; do
    python configs/generate_configs.py --base_path $BASE --version $VERSION --dataset $d --models $MODEL \
        --tokens $TOKENS --lrs $LR --bsz 16 --vocab_size $VOCAB --max_seq_length 1024 --seeds 0 --min_tokens_per_param 0 \
        --run_prefix ${RUN_PREFIX}_${d}_ --wandb_project open_optformer_gp_gonogo --out_dir $CONFIG_DIR > /dev/null
  done
}

case "${1:-}" in
  rank)
    python generate_training_data/rank_gp_teachers.py --corpus $CORPUS --output_path $OUT/ranks --workers $WORKERS ;;
  export)
    for d in $DATASETS; do
      [[ -n "${ARGS[$d]:-}" ]] || { echo "[FATAL] unknown dataset $d (known: ${!ARGS[*]})" >&2; exit 1; }
      $EXPORT --output_path $OUT/export/$d ${ARGS[$d]} && shuffle $OUT/export/$d
    done ;;
  tokenizer)
    # one tokenizer for all GP experiments: a small function sample of every scheme (incl. hp suffix, symmetries,
    # shorter trajectories); the vocabulary is symbols + characters only (sentencepiece: min = max), 1062 on the pilot
    T=$OUT/tok_exports; S="--frac_functions 0.005 --symmetries 8 --sample_shorter_trajectories"
    for sc in generic kernel kernel_mode quality best; do $EXPORT --output_path $T/$sc --tag_scheme $sc --tag_dropout 0.2 $S; done
    $EXPORT --output_path $T/kernel_hp --tag_scheme kernel --hp_suffix $S
    $EXPORT --output_path $T/kernel_mode_hp --tag_scheme kernel_mode --hp_suffix --tag_dropout 0.2 $S
    python generate_training_data/make_tokenizer_sample.py --exports $T/* --n_lines 50000 --output $OUT/tok_sample.txt
    python generate_training_data/train_tokenizer.py --input_files $OUT/tok_sample.txt --output_path $BASE/tokenizer/$VERSION \
        --vocab_size $VOCAB --input_sentence_size 50000 --extra_symbols generate_training_data/gp_tokenizer_symbols.txt
    python generate_training_data/check_gp_tokenizer.py --tokenizer_dir $BASE/tokenizer/$VERSION \
        --symbols generate_training_data/gp_tokenizer_symbols.txt --exports $T/* | tee $OUT/tokenizer_check.json ;;
  preprocess)
    # litdata stages its item list (pickle) and chunks in $TMPDIR / its cache folders before moving them to the
    # output; node-local /tmp is too small for R (9.4M lines), so keep everything on the workspace and clean up after
    for d in $DATASETS; do
      SCRATCH=$OUT/tmp/preprocess_${d}_${SLURM_JOB_ID:-$$}
      mkdir -p $SCRATCH/{tmp,chunks,data}
      trap 'rm -rf "$SCRATCH"' EXIT
      export TMPDIR=$SCRATCH/tmp DATA_OPTIMIZER_CACHE_FOLDER=$SCRATCH/chunks DATA_OPTIMIZER_DATA_CACHE_FOLDER=$SCRATCH/data
      (cd generate_training_data && python preprocess_data.py --input_path $OUT/export/$d \
          --output_path $BASE/tokenized_dataset/$VERSION/$d --tokenizer_dir $BASE/tokenizer/$VERSION --block_size 1025)
      for s in train valid; do
        [[ -f $BASE/tokenized_dataset/$VERSION/$d/$s/index.json ]] || { echo "[FATAL] preprocess $d/$s wrote no index.json" >&2; exit 1; }
      done
      echo "[INFO] preprocess $d done"
    done ;;
  configs)
    make_configs ;;
  train)
    make_configs
    LOGGER=()
    [[ -n "${WANDB_API_KEY:-}" ]] || { echo "[INFO] WANDB_API_KEY not set, logging to csv"; LOGGER=(--logger_name csv); }
    for d in $DATASETS; do
      cfg=$(config_of $d)
      [[ -n "$cfg" ]] || { echo "[FATAL] no config for $d in $CONFIG_DIR" >&2; exit 1; }
      for s in train valid; do
        [[ -f $BASE/tokenized_dataset/$VERSION/$d/$s/index.json ]] || {
          echo "[FATAL] $BASE/tokenized_dataset/$VERSION/$d/$s has no index.json: preprocess for $d has not finished (or failed)" >&2; exit 1; }
      done
      echo "[INFO] training $d with $cfg"
      python open_optformer/training/pretrain.py "Qwen3-${MODEL#qwen3_}" --config "$cfg" "${LOGGER[@]}"
    done ;;
  rollouts)
    for d in $DATASETS; do
      python benchmarks/gp_functions/run_gp_benchmark.py --corpus $CORPUS --checkpoint_dir "$(ckpt_of $d)" \
          --tags "${TAGS[@]}" --subset test_core --seeds 0 1 2 3 4 --shard $SHARD --y_norm "$(ynorm_of $d)" \
          --temperature $TEMPERATURE --output $OUT/rollouts/$(label_of $d)_shard_${SHARD/\//of}.jsonl
    done ;;
  score)
    for d in ${SCORE_DATASETS:-R}; do
      python benchmarks/gp_functions/score_teacher_traces.py --corpus $CORPUS --checkpoint_dir "$(ckpt_of $d)" \
          --subset test_core --seeds 0 1 2 3 4 --y_norm "$(ynorm_of $d)" --output $OUT/scores_$(TEMPERATURE=1 label_of $d).parquet
    done ;;
  analyze)
    # LABELS: rollout labels to report on (default: $DATASETS); the first is the reference of the arm comparison
    LABELS=${LABELS:-$DATASETS}
    arms=()
    for l in $LABELS; do
      shopt -s nullglob; ro=($OUT/rollouts/${l}_shard_*.jsonl); shopt -u nullglob
      [[ ${#ro[@]} -gt 0 ]] || { echo "[WARN] no rollouts for $l, skipped"; continue; }
      sc=(); [[ -f $OUT/scores_$l.parquet ]] && sc=(--scores $OUT/scores_$l.parquet)
      python benchmarks/gp_functions/analyze_gp_eval.py --corpus $CORPUS --rollouts "${ro[@]}" "${sc[@]}" \
          --output_path $OUT/eval/$l
      arms+=("$l=$OUT/rollouts/${l}_shard_*.jsonl")
    done
    name=${REPORT_NAME:-$(echo $LABELS | tr ' ' '_')}
    if [[ ${#arms[@]} -ge 2 ]]; then
      python benchmarks/gp_functions/compare_gp_arms.py --corpus $CORPUS --arms "${arms[@]}" \
          --output_path $OUT/compare/$name
    fi
    [[ ${#arms[@]} -ge 1 ]] && python benchmarks/gp_functions/plot_gp_rollouts.py --corpus $CORPUS --arms "${arms[@]}" \
        --tags ${PLOT_TAGS:-GP-\{K\}-ORACLE} --output_path $OUT/plots/$name ;;
  *)
    sed -n '2,23p' "$0"; exit 1 ;;
esac
