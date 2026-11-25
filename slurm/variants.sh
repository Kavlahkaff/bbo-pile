# The runs: "variant model tokens lr seed" per line. RUNS="..." (same format, separated by ;) selects others.
# log         log y encoding, tags GP NM RS BEST
# log_marker  the same, with the end of the initial design marked
DEFAULT_RUNS="log qwen3_5M 2.4B 5e-3 0;log qwen3_5M 2.4B 5e-3 1;log qwen3_5M 2.4B 5e-3 2
log_marker qwen3_5M 2.4B 5e-3 0;log_marker qwen3_5M 2.4B 5e-3 1;log_marker qwen3_5M 2.4B 5e-3 2
log qwen3_30M 2.4B 1e-3 0;log qwen3_30M 2.4B 2e-3 0
log_marker qwen3_30M 2.4B 1e-3 0;log_marker qwen3_30M 2.4B 2e-3 0
log qwen3_80M 2.4B 5e-4 0;log qwen3_80M 2.4B 1e-3 0
log_marker qwen3_80M 2.4B 5e-4 0;log_marker qwen3_80M 2.4B 1e-3 0"
runs() { tr ';' '\n' <<< "${RUNS:-$DEFAULT_RUNS}" | sed '/^ *$/d'; }
use() {   # use <variant> <model> <tokens> <lr> <seed>: exports the settings and sets RUN, NAME, ...
  case "$1" in log) export MARKER=0 ;; log_marker) export MARKER=1 ;; *) echo "unknown variant $1" >&2; return 1 ;; esac
  [[ $# == 5 ]] || { echo "a run needs: variant model tokens lr seed (got: $*)" >&2; return 1; }
  export MODEL=$2 TOKENS=$3 LR=$4 MODEL_SEED=$5
  source slurm/env.sh
}
label() { echo "$1_${2}_${3}_lr$4"; }   # run settings without the seed, for compare and match
