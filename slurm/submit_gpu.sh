#!/bin/bash
# Capella login node, repo root, after prepare: train -> rollouts. A run that has results (or was submitted) is skipped, so
# nothing is overwritten. DRY=1 only lists. RUNS="log_marker qwen3_5M 2.4B 5e-3 0;..." selects runs.
set -euo pipefail
[[ -f slurm/train.sbatch ]] || { echo "run from the repo root" >&2; exit 1; }
source slurm/variants.sh
unset TEMP TAGS
mkdir -p logs
checked=
while read -r -a r; do
  use "${r[@]}"
  found=
  for p in "$CKPT" "$REPORTS/$RUN.txt" "$WORK/submitted/$RUN" "$ROLLOUTS/${RUN}"_[0-9]*.jsonl; do
    if [[ -e $p ]]; then found=$p; break; fi
  done
  if [[ -n $found ]]; then echo "skipped $RUN: exists $found"; continue; fi
  for s in train valid; do [[ -f $DATA/$s/index.json ]] || { echo "no $DATA/$s/index.json, run prepare first" >&2; exit 1; }; done
  if [[ $checked != *" $NAME "* && ${LITDATA_POSIX_FAST:-1} != 0 ]]; then
    # litdata gives each of the 8 loader workers whole chunks and cuts all to the smallest; the share depends on the chunk sizes
    python3 - "$DATA/train/index.json" <<'PY' || { echo "nothing more submitted: see the share of $NAME above. Tokenize again, or set LITDATA_POSIX_FAST=0" >&2; exit 1; }
import json, sys
sizes = [c["dim"] // 1025 for c in json.load(open(sys.argv[1]))["chunks"]]
total, w, prefix, load, n = sum(sizes), 0, 0, [0] * 8, [0] * 8
for x in sizes:
    while w < 7 and n[w] and prefix + x / 2 > (w + 1) * total / 8:
        w += 1
    load[w] += x; n[w] += 1; prefix += x
share = min(v for v in load if v) // 16 * 16 * sum(v > 0 for v in load) / total
print(f"the loader will use {share:.1%} of the train data")
sys.exit(share < 0.9)
PY
    checked+=" $NAME "
  fi
  if [[ ${DRY:-0} == 1 ]]; then echo "would submit $RUN"; continue; fi
  mkdir -p "$WORK/submitted"; mkdir "$WORK/submitted/$RUN"       # blocks a second submission
  case "$MODEL" in qwen3_80M) limit=08:00:00 ;; *) limit=04:00:00 ;; esac      # 30M needs 1.75 h, 80M an estimated 4-5 h
  t=$(sbatch --parsable --export=ALL --time="$limit" --job-name="sel-train-$RUN" slurm/train.sbatch </dev/null); t=${t%%;*}
  b=$(sbatch --parsable --export=ALL --job-name="sel-rollouts-$RUN" --kill-on-invalid-dep=yes --dependency=afterok:"$t" slurm/rollouts.sbatch </dev/null); b=${b%%;*}
  echo "$RUN: train=$t rollouts=$b"
done < <(runs)
