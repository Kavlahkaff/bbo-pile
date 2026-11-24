#!/bin/bash
#SBATCH --job-name=sel-pilot-report
#SBATCH --account=p_neurasearch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Barnard. Completeness, rule and verdict -> $PILOT/report.txt
source slurm/common.sh cpu
python -c "import sklearn" || { echo "[FATAL] scikit-learn missing in $VENV" >&2; exit 1; }
python -m datagen.pilot "$PILOT" --funcs "$PILOT_FUNCS" | tee "$PILOT/report.txt"
