#!/bin/bash
# bash submit_exp.sh 1|2|3 (Capella: train -> rollouts), bash submit_exp.sh report 1 [2 3] (Barnard). Run from the repo root.
set -euo pipefail
[[ -f slurm/train.sbatch ]] || { echo "run from the repo root" >&2; exit 1; }
unset NAME TAGS MODEL TOKENS LR MODEL_SEED CORPUS SEEDS
source slurm/variants.sh
export WORK=${WORK:-/data/horse/ws/luth474h-fmbo/optimizer_selection_v3}
SRC=ALL_BEST_causal_log_noUNK
mkdir -p logs
runs_of() {   # model tokens lr seed train_time
  case "$1" in
    1) echo "qwen3_5M 2.4B 2e-3 0 04:00:00"; echo "qwen3_5M 2.4B 2e-3 1 04:00:00" ;;
    2) echo "qwen3_5M 4.8B 5e-3 0 06:00:00" ;;
    3) echo "qwen3_30M 2.4B 1e-3 0 12:00:00"; echo "qwen3_30M 2.4B 2e-3 0 12:00:00" ;;
    *) echo "unknown experiment $1" >&2; return 1 ;;
  esac
}
setup() {     # same data as noUNK; the learning rate goes into NAME, so RUN is new
  variant noUNK
  export MODEL=$1 TOKENS=$2 LR=$3 MODEL_SEED=$4
  export NAME=${SRC}_lr$LR
  RUN=${NAME}_${MODEL}_${TOKENS}; [[ $MODEL_SEED == 0 ]] || RUN+=_s$MODEL_SEED
}
if [[ ${1:-} == report ]]; then
  shift; ARGS="--base noUNK"
  for s in "" _s1 _s2; do ARGS+=" --run noUNK=$WORK/rollouts/${SRC}_qwen3_5M_2.4B$s"; done
  for e in "$@"; do
    R=$(runs_of "$e")
    while read -r m t l s _; do
      setup "$m" "$t" "$l" "$s"
      [[ ! -e $WORK/reports/$RUN.txt ]] || { echo "report exists: $RUN" >&2; exit 1; }
      echo "$RUN: $(sbatch --parsable --export=ALL --job-name="sel-analyze-$RUN" slurm/analyze.sbatch </dev/null)"
      ARGS+=" --run ${m}_${t}_lr$l=$WORK/rollouts/$RUN"
    done <<< "$R"
  done
  export ARGS OUT=$WORK/reports/compare_exp_$(date +%Y%m%d_%H%M%S).txt
  sbatch --export=ALL <<'EOF'
#!/bin/bash
#SBATCH --job-name=sel-compare-exp
#SBATCH --account=p_neurasearch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
source slurm/common.sh cpu
python -m evaluation.compare $ARGS --output "$OUT"
EOF
  exit
fi
R=$(runs_of "${1:-}")
T=$WORK/tokenized_dataset/selection_v1
[[ -f $T/$SRC/train/index.json ]] || { echo "no data at $T/$SRC" >&2; exit 1; }
while read -r m t l s time; do       # check everything before submitting anything
  setup "$m" "$t" "$l" "$s"
  for p in "$WORK/checkpoints/selection_v1/$RUN" "$WORK/reports/$RUN.txt" "$WORK/submitted/$RUN" "$WORK/rollouts/${RUN}"_[0-9]*.jsonl; do
    [[ ! -e $p ]] || { echo "exists, nothing submitted: $p" >&2; exit 1; }
  done
done <<< "$R"
mkdir -p "$WORK/submitted"
while read -r m t l s time; do
  setup "$m" "$t" "$l" "$s"
  [[ -e $T/$NAME ]] || ln -s "$SRC" "$T/$NAME"
  [[ $(readlink "$T/$NAME") == "$SRC" ]] || { echo "$T/$NAME is not the expected link" >&2; exit 1; }
  mkdir "$WORK/submitted/$RUN"       # blocks a second submission
  a=$(sbatch --parsable --export=ALL --time="$time" --job-name="sel-train-$RUN" slurm/train.sbatch </dev/null); a=${a%%;*}
  b=$(sbatch --parsable --export=ALL --time=08:00:00 --job-name="sel-rollouts-$RUN" --kill-on-invalid-dep=yes --dependency=afterok:"$a" slurm/rollouts.sbatch </dev/null); b=${b%%;*}
  echo "$RUN: train=$a rollouts=$b"
done <<< "$R"
