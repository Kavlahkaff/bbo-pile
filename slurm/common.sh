# source slurm/common.sh cpu|gpu   (jobs are submitted from the repo root)
set -euo pipefail
trap 'echo "[FATAL] line $LINENO: $BASH_COMMAND" >&2' ERR
REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
[[ -f slurm/env.sh && -d datagen ]] || { echo "[FATAL] submit from the repo root (got $REPO)" >&2; exit 1; }
source slurm/env.sh
case "$1" in
  cpu) MODULES=$MODULES_CPU; VENV=$VENV_CPU ;;
  gpu) MODULES=$MODULES_GPU; VENV=$VENV_GPU ;;
esac
module load $MODULES
[[ -f $VENV/bin/activate ]] || { echo "[FATAL] no venv $VENV (set it in slurm/env.sh)" >&2; exit 1; }
source "$VENV/bin/activate"
if [[ $1 == gpu ]] && ! nvidia-smi &>/dev/null; then echo "[FATAL] no GPU visible (submit from Capella)" >&2; exit 1; fi
if [[ -f $WANDB_ENV_FILE ]]; then set -a; source "$WANDB_ENV_FILE"; set +a; fi
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$WORK"
