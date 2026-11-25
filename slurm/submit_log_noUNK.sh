#!/bin/bash
# log + noUNK, three model seeds; from the repo root, never overwrites a result.
#   bash slurm/submit_log_noUNK.sh prepare   Barnard: export + tokenize
#   bash slurm/submit_log_noUNK.sh train     Capella, when prepare is done: train -> rollouts
#   bash slurm/submit_log_noUNK.sh report    Barnard, when the rollouts are done: reports and a new compare file
set -euo pipefail
[[ -f slurm/train.sbatch ]] || { echo "run from the repo root" >&2; exit 1; }
unset NAME TAGS MODEL TOKENS LR MODEL_SEED CORPUS SEEDS
source slurm/variants.sh
export WORK=${WORK:-/data/horse/ws/luth474h-fmbo/optimizer_selection_v3}
variant log_noUNK
export NAME=ALL_BEST_log_noUNK MODEL=qwen3_5M TOKENS=2.4B LR=5e-3
DATA=$WORK/tokenized_dataset/selection_v1/$NAME
SEED_LIST=$(seeds_of log_noUNK)
mkdir -p logs
run_of() { echo "${NAME}_${MODEL}_${TOKENS}$([[ $1 == 0 ]] || echo "_s$1")"; }
absent() { for p in "$@"; do [[ ! -e $p ]] || { echo "exists, nothing submitted: $p" >&2; exit 1; }; done; }

case "${1:-}" in
prepare)
  absent "$DATA" "$WORK/export/$NAME" "$WORK/submitted/prepare_$NAME"
  mkdir -p "$WORK/submitted"; mkdir "$WORK/submitted/prepare_$NAME"   # blocks a second submission
  echo "prepare: $(sbatch --parsable --export=ALL --job-name=sel-prepare-log_noUNK slurm/prepare.sbatch)" ;;
train)
  for s in train valid; do [[ -f $DATA/$s/index.json ]] || { echo "no $DATA/$s/index.json, run prepare first" >&2; exit 1; }; done
  # litdata gives each of the 8 loader workers whole chunks and cuts all to the smallest; the share depends on the chunk sizes
  if [[ ${LITDATA_POSIX_FAST:-1} != 0 ]]; then
    python3 - "$DATA/train/index.json" <<'PY'
import json, sys
sizes = [c["dim"] // 1025 for c in json.load(open(sys.argv[1]))["chunks"]]
total, w, prefix, load, n = sum(sizes), 0, 0, [0] * 8, [0] * 8
for x in sizes:
    while w < 7 and n[w] and prefix + x / 2 > (w + 1) * total / 8:
        w += 1
    load[w] += x; n[w] += 1; prefix += x
share = min(v for v in load if v) // 16 * 16 * sum(v > 0 for v in load) / total
print(f"the loader will use {share:.1%} of the train data (about 94% in noUNK and log)")
sys.exit(share < 0.9)
PY
  fi || { echo "nothing submitted: see the share above. Export and tokenize again, or set LITDATA_POSIX_FAST=0" >&2; exit 1; }
  for s in $SEED_LIST; do
    r=$(run_of "$s")
    absent "$WORK/checkpoints/selection_v1/$r" "$WORK/reports/$r.txt" "$WORK/submitted/$r" "$WORK/rollouts/${r}"_[0-9]*.jsonl
  done
  for s in $SEED_LIST; do
    export MODEL_SEED=$s; r=$(run_of "$s")
    mkdir -p "$WORK/submitted"; mkdir "$WORK/submitted/$r"
    t=$(sbatch --parsable --export=ALL --job-name="sel-train-log_noUNK-s$s" slurm/train.sbatch); t=${t%%;*}
    b=$(sbatch --parsable --export=ALL --job-name="sel-rollouts-log_noUNK-s$s" --kill-on-invalid-dep=yes --dependency=afterok:"$t" slurm/rollouts.sbatch); b=${b%%;*}
    echo "$r: train=$t rollouts=$b"
  done ;;
report)
  ARGS="--base noUNK"
  for s in "" _s1 _s2; do ARGS+=" --run noUNK=$WORK/rollouts/ALL_BEST_causal_log_noUNK_qwen3_5M_2.4B$s"; done
  ARGS+=" --run log=$WORK/rollouts/ALL_BEST_log_qwen3_5M_2.4B"
  for s in $SEED_LIST; do absent "$WORK/reports/$(run_of "$s").txt"; done
  for s in $SEED_LIST; do
    export MODEL_SEED=$s
    echo "$(run_of "$s"): $(sbatch --parsable --export=ALL --job-name="sel-analyze-log_noUNK-s$s" slurm/analyze.sbatch)"
    ARGS+=" --run log_noUNK=$WORK/rollouts/$(run_of "$s")"
  done
  export ARGS OUT=$WORK/reports/compare_log_noUNK_$(date +%Y%m%d_%H%M%S).txt
  sbatch --export=ALL <<'EOF'
#!/bin/bash
#SBATCH --job-name=sel-compare-log_noUNK
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
  ;;
*) echo "usage: bash slurm/submit_log_noUNK.sh prepare|train|report" >&2; exit 1 ;;
esac
