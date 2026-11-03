# Paths and environments for the GP go/no-go jobs. Edit here only; every job sources this file.
# BASE must be readable/writable from both Barnard (CPU jobs) and Capella (GPU jobs).
CORPUS=/data/horse/ws/luth474h-fmbo/fmbo-data-gen/corpus_student_40k   # several dirs: space separated
BASE=/data/horse/ws/luth474h-fmbo/bbo-pile_experiments                         # tokenizer/, tokenized_dataset/, checkpoints/
OUT=/data/horse/ws/luth474h-fmbo/gp_gonogo                                # exports, ranks, rollouts, reports

# Barnard (CPU): export, tokenizer, preprocessing, ranking, analysis
MODULES_CPU="release/2026 GCCcore/14.2.0 Python/3.13.1"
VENV_CPU=$REPO/.venv-datagen
# Capella (GPU): training, rollouts, teacher-forced scoring
MODULES_GPU="release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0"
VENV_GPU=$REPO/.venv-litgpt

# training run of the go/no-go
TOKENS=300M
LR=5e-3

# optional: file with WANDB_API_KEY=... (without it training logs to csv)
WANDB_ENV_FILE=$REPO/.env.wandb
