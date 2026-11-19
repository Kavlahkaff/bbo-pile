#!/bin/bash
#SBATCH --job-name=gp-contrast
#SBATCH --account=p_neurasearch
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-49
#SBATCH --output=logs/gp/%x-%A_%a.out
# Barnard. One shard of a contrast screen (figures/09_contrast); no model needed. Each array task runs every
# ${NSHARDS}-th (regime, function) job into figures/09_contrast/raw/<OUT_NAME>.shard<k>.jsonl (resumable: resubmitting
# skips finished jobs). Combine afterwards with
#   (cd figures/09_contrast/src && python -c "import common; common.merge('<OUT_NAME>')")
#
# Default = screen A at scale (2026-11-19 notes: 400 functions x 30 seeds, ~52 CPU-h):
#   MASTER_THESIS_DIR=/path/to/master-thesisv2 sbatch slurm/gp/contrast.sbatch
# Other screens: SCREEN=screen_f FUNCS=160 SEEDS=5 OUT_NAME=screen_f TEACHERS=all sbatch --array=0-9 ...
# (set NSHARDS to the array size when it is not 50).
source slurm/gp/common.sh cpu

: "${MASTER_THESIS_DIR:?set MASTER_THESIS_DIR to the master-thesisv2 checkout that contains gp_traj.py}"
[[ -f $MASTER_THESIS_DIR/gp_traj.py ]] || { echo "[FATAL] no gp_traj.py in $MASTER_THESIS_DIR" >&2; exit 1; }
export MASTER_THESIS_DIR
python -c "import sklearn, scipy" || { echo "[FATAL] venv $VENV needs scikit-learn and scipy" >&2; exit 1; }

SCREEN=${SCREEN:-screen_a}
FUNCS=${FUNCS:-400}
SEEDS=${SEEDS:-30}
OUT_NAME=${OUT_NAME:-screen_a_big}
TEACHERS=${TEACHERS:-rs,m32,et_ei,et_dist0.5,smac3_trees}
NSHARDS=${NSHARDS:-50}
TASK=${SLURM_ARRAY_TASK_ID:?submit as an array job}

ARGS=("$FUNCS" "$SEEDS" "$OUT_NAME")
[[ $TEACHERS != all ]] && ARGS+=("$TEACHERS")
cd figures/09_contrast/src
python "$SCREEN.py" "${ARGS[@]}" --shard "$TASK/$NSHARDS" --workers "${SLURM_CPUS_PER_TASK:-4}"
