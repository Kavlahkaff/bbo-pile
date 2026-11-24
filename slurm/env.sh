# Paths and environments; edit here only. WORK must be reachable from Barnard (CPU) and Capella (GPU).
WORK=${WORK:-/data/horse/ws/luth474h-fmbo/optimizer_selection_v2}   # v1 = gp + rosen only, different task layout: never mix
CORPUS=${CORPUS:-$WORK/corpus}
MODULES_CPU="release/2026 GCCcore/14.2.0 Python/3.13.1"
MODULES_GPU="release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0"
VENV_CPU=${VENV_CPU:-/data/horse/ws/luth474h-fmbo/bbo-pile/.venv-datagen}
VENV_GPU=${VENV_GPU:-/data/horse/ws/luth474h-fmbo/bbo-pile/.venv-litgpt}
WANDB_ENV_FILE=${WANDB_ENV_FILE:-$REPO/.env.wandb}
# experiment (override per submission, e.g. MODEL=qwen3_10M bash slurm/submit_gpu.sh)
TRAIN_FUNCS=${TRAIN_FUNCS:-49998}      # gp functions; each other trained family gets a third
PILOT=${PILOT:-$WORK/pilot}
PILOT_FUNCS=${PILOT_FUNCS:-1000}       # per family
DATASET=${DATASET:-ALL_BEST}           # ALL, BEST, ALL_BEST
Y_NORM=${Y_NORM:-causal_log}           # causal_log, log, global, causal
N_PREFIXES=${N_PREFIXES:-0}            # 2 for the rescaled encodings (log, global)
MODEL=${MODEL:-qwen3_5M}
TOKENS=${TOKENS:-1.2B}
LR=${LR:-5e-3}
EXCLUDE=${EXCLUDE:-}                   # optimizers left out of the training data, e.g. RS
if [[ -z "${TAGS:-}" ]]; then          # tags for the rollouts: what the model was trained on
  TAGS=""
  for o in GP NM RS; do [[ " $EXCLUDE " == *" $o "* ]] || TAGS+="$o "; done
  TAGS+="BEST UNK"
fi
if [[ -z "${NAME:-}" ]]; then          # data set name; the model is ${NAME}_${MODEL}_${TOKENS}
  NAME=${DATASET}${EXCLUDE:+_no${EXCLUDE// /-}}_${Y_NORM}
  if [[ $N_PREFIXES != 0 ]]; then NAME+=_p$N_PREFIXES; fi
fi
