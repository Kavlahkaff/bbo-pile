# Paths and environments; edit here only. WORK must be reachable from Barnard (CPU) and Capella (GPU).
WORK=${WORK:-/data/horse/ws/luth474h-fmbo/optimizer_selection_v3}
CORPUS=${CORPUS:-$WORK/corpus}
MODULES_CPU="release/2026 GCCcore/14.2.0 Python/3.13.1"
MODULES_GPU="release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0"
VENV_CPU=${VENV_CPU:-/data/horse/ws/luth474h-fmbo/bbo-pile/.venv-datagen}
VENV_GPU=${VENV_GPU:-/data/horse/ws/luth474h-fmbo/bbo-pile/.venv-litgpt}
WANDB_ENV_FILE=${WANDB_ENV_FILE:-${REPO:-$PWD}/.env.wandb}
TRAIN_FUNCS=${TRAIN_FUNCS:-99996}      # gp functions; each other trained family gets a third
# experiment; normally set by slurm/variants.sh
MARKER=${MARKER:-0}                    # 1: the end of the initial design is marked in the prompt
MODEL=${MODEL:-qwen3_5M}
TOKENS=${TOKENS:-2.4B}
LR=${LR:-5e-3}
MODEL_SEED=${MODEL_SEED:-0}            # initialisation and data order of the training run
TEMP=${TEMP:-1}                        # sampling temperature of the rollouts; other values get their own directories
TAGS=${TAGS:-"GP NM RS BEST"}
# names on disk: data set NAME, model RUN. "noUNK" is kept from the earlier round, so its results are found
NAME=ALL_BEST_log_noUNK
TOK=$WORK/tokenizer SYMBOLS=generate_training_data/tokenizer_symbols.txt VOCAB_FILE=generate_training_data/vocab_size.txt
MARKER_ARG=()
if [[ $MARKER == 1 ]]; then            # the marker needs its own tokenizer; the other one must not change
  NAME+=_marker TOK+=_marker SYMBOLS=${SYMBOLS%.txt}_marker.txt VOCAB_FILE=${VOCAB_FILE%.txt}_marker.txt
  MARKER_ARG=(--marker)
fi
DATA=$WORK/tokenized_dataset/selection_v1/$NAME
RUN=${NAME}_${MODEL}_${TOKENS}
if [[ $LR != 5e-3 ]]; then RUN+=_lr$LR; fi
if [[ $MODEL_SEED != 0 ]]; then RUN+=_s$MODEL_SEED; fi
CKPT=$WORK/checkpoints/selection_v1/$RUN
SUFFIX=$([[ $TEMP == 1 ]] || echo "_T$TEMP")
ROLLOUTS=$WORK/rollouts$SUFFIX REPORTS=$WORK/reports$SUFFIX
