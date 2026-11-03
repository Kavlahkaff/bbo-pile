#!/usr/bin/env bash
# Early go/no-go check (before building the full experiment grid):
#   export O and R (no symmetries), train a 5M model on each, then on test_core check
#   (a) the student prompted as the oracle vs the matched-kernel MLE teacher, floored at 1e-3 (rollouts)
#   (b) whether the tag-likelihood matrix separates oracle / MLE / RS (teacher-forced scores, model R)
#
# Usage: bash benchmarks/gp_functions/gonogo.sh <stage>     stages, in order:
#   rank        step 0 teacher ranking (CPU, ~tens of minutes)                     -> $OUT/ranks
#   export      exports R and O, shuffled                                         -> $OUT/export/{R,O}
#   tokenizer   small exports of every tag scheme -> mixed sample -> gp_v1 + checks -> $BASE/tokenizer/gp_v1
#   preprocess  tokenize R and O into litdata chunks                               -> $BASE/tokenized_dataset/gp_v1/{R,O}
#   configs     training configs (5M, $TOKENS, lr $LRS, bsz 16)                    -> configs/gp_gonogo/*.yaml
#               then train each:  python open_optformer/training/pretrain.py Qwen3-5M --config <yaml>
#   rollouts    student rollouts on test_core x 5 seeds (CPU; use SHARD=i/n in a SLURM array)
#   score       teacher-forced tag-likelihood scores of model R on test_core (GPU or CPU)
#   analyze     report + verdict.json                                              -> $OUT/eval/{R,O}
#
# Environment (defaults for Barnard):
#   CORPUS  corpus directory (or several, space separated)
#   BASE    bbo-pile experiments root (tokenizer/, tokenized_dataset/, checkpoints/)
#   OUT     working directory for exports, ranks, rollouts, reports
#   CKPT_R, CKPT_O   trained checkpoints (.../final) for rollouts / score / analyze
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CORPUS=${CORPUS:-/data/horse/ws/luth474h-fmbo/fmbo-data-gen/corpus_student_40k}
BASE=${BASE:-/projects/p_neurasearch/bbo-pile_experiments}
OUT=${OUT:-/data/horse/ws/luth474h-fmbo/gp_gonogo}
VERSION=gp_v1
TOKENS=${TOKENS:-300M}
LRS=${LRS:-5e-3}
WORKERS=${WORKERS:-$(nproc)}
SHARD=${SHARD:-0/1}
TAGS=('GP-{K}-ORACLE' 'GP-{K}-MLE' 'RS' 'UNK' 'RS-baseline')
EXPORT="python generate_training_data/compile_gp_data.py --corpus $CORPUS --workers $WORKERS"
# reference R: kernel_mode tags, all teachers, tag dropout 0.2, shorter trajectories, no symmetries (S1)
R_ARGS="--tag_scheme kernel_mode --teachers all --tag_dropout 0.2 --sample_shorter_trajectories"
O_ARGS="--tag_scheme kernel_mode --teachers oracle --tag_dropout 0.2 --sample_shorter_trajectories"

shuffle() { for s in train valid; do shuf "$1/$s.txt" -o "$1/$s.txt"; done; }

case "${1:-}" in
  rank)
    python generate_training_data/rank_gp_teachers.py --corpus $CORPUS --output_path $OUT/ranks --workers $WORKERS ;;
  export)
    $EXPORT --output_path $OUT/export/R $R_ARGS && shuffle $OUT/export/R
    $EXPORT --output_path $OUT/export/O $O_ARGS && shuffle $OUT/export/O ;;
  tokenizer)
    # one tokenizer for all GP experiments: a 2% function sample of every scheme (incl. hp suffix, symmetries, shorter)
    T=$OUT/tok_exports; S="--frac_functions 0.02 --symmetries 8 --sample_shorter_trajectories"
    for sc in generic kernel kernel_mode quality best; do $EXPORT --output_path $T/$sc --tag_scheme $sc --tag_dropout 0.2 $S; done
    $EXPORT --output_path $T/kernel_hp --tag_scheme kernel --hp_suffix $S
    $EXPORT --output_path $T/kernel_mode_hp --tag_scheme kernel_mode --hp_suffix --tag_dropout 0.2 $S
    python generate_training_data/make_tokenizer_sample.py --exports $T/* --n_lines 50000 --output $OUT/tok_sample.txt
    # the vocabulary is symbols + characters only (sentencepiece: min = max); 1062 on the pilot
    python generate_training_data/train_tokenizer.py --input_files $OUT/tok_sample.txt --output_path $BASE/tokenizer/$VERSION \
        --vocab_size ${VOCAB:-1062} --input_sentence_size 50000 --extra_symbols generate_training_data/gp_tokenizer_symbols.txt
    python generate_training_data/check_gp_tokenizer.py --tokenizer_dir $BASE/tokenizer/$VERSION \
        --symbols generate_training_data/gp_tokenizer_symbols.txt --exports $T/* | tee $OUT/tokenizer_check.json ;;
  preprocess)
    for d in R O; do
      (cd generate_training_data && python preprocess_data.py --input_path $OUT/export/$d \
          --output_path $BASE/tokenized_dataset/$VERSION/$d --tokenizer_dir $BASE/tokenizer/$VERSION --block_size 1025)
    done ;;
  configs)
    for d in R O; do
      python configs/generate_configs.py --base_path $BASE --version $VERSION --dataset $d --models qwen3_5M \
          --tokens $TOKENS --lrs $LRS --bsz 16 --vocab_size ${VOCAB:-1062} --max_seq_length 1024 --seeds 0 \
          --run_prefix gonogo_${d}_ --wandb_project open_optformer_gp_gonogo --out_dir configs/gp_gonogo
    done ;;
  rollouts)
    for m in R O; do
      ck=CKPT_$m
      python benchmarks/gp_functions/run_gp_benchmark.py --corpus $CORPUS --checkpoint_dir ${!ck} \
          --tags "${TAGS[@]}" --subset test_core --seeds 0 1 2 3 4 --shard $SHARD \
          --output $OUT/rollouts/${m}_shard_${SHARD/\//of}.jsonl
    done ;;
  score)
    python benchmarks/gp_functions/score_teacher_traces.py --corpus $CORPUS --checkpoint_dir $CKPT_R \
        --subset test_core --seeds 0 1 2 3 4 --output $OUT/scores_R.parquet ;;
  analyze)
    python benchmarks/gp_functions/analyze_gp_eval.py --corpus $CORPUS --rollouts $OUT/rollouts/R_shard_*.jsonl \
        --scores $OUT/scores_R.parquet --output_path $OUT/eval/R
    python benchmarks/gp_functions/analyze_gp_eval.py --corpus $CORPUS --rollouts $OUT/rollouts/O_shard_*.jsonl \
        --output_path $OUT/eval/O ;;
  *)
    sed -n '2,24p' "$0"; exit 1 ;;
esac
