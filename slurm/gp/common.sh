# Sourced by every job in slurm/gp/: `source slurm/gp/common.sh cpu|gpu`.
# Jobs must be submitted from the repo root (sbatch slurm/gp/<job>.sbatch), so $SLURM_SUBMIT_DIR is the repo.
set -euo pipefail
trap 'echo "[FATAL] line $LINENO: command failed: $BASH_COMMAND" >&2' ERR

REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
if [[ ! -f benchmarks/gp_functions/gonogo.sh ]]; then
  echo "[FATAL] submit from the bbo-pile repo root (got $REPO)" >&2
  exit 1
fi
source "${GP_ENV_FILE:-slurm/gp/env.sh}"

case "${1:?usage: source slurm/gp/common.sh cpu|gpu}" in
  cpu) MODULES=$MODULES_CPU; VENV=$VENV_CPU ;;
  gpu) MODULES=$MODULES_GPU; VENV=$VENV_GPU ;;
  *) echo "[FATAL] unknown kind $1" >&2; exit 1 ;;
esac
module load $MODULES
if [[ ! -f $VENV/bin/activate ]]; then
  echo "[FATAL] venv $VENV not found -- create it as described in slurm/gp/README.md" >&2
  exit 1
fi
source $VENV/bin/activate

if [[ "$1" == gpu ]] && ! nvidia-smi &>/dev/null; then
  echo "[FATAL] no GPU visible to this job (submit GPU jobs from a Capella login node)" >&2
  exit 1
fi
if [[ -f "${WANDB_ENV_FILE:-}" ]]; then
  set -a; source "$WANDB_ENV_FILE"; set +a
fi

export CORPUS BASE OUT TOKENS LR
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS=ignore
mkdir -p "$OUT"

# array jobs over datasets: task 0 -> R, task 1 -> O
DATASET_LIST=(R O)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -z "${DATASETS:-}" && "${ARRAY_OVER_DATASETS:-0}" == 1 ]]; then
  export DATASETS=${DATASET_LIST[$SLURM_ARRAY_TASK_ID]}
fi
echo "[INFO] $(date) job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:--} on $(hostname): venv=$VENV" \
     "DATASETS=${DATASETS:-all} OUT=$OUT"
