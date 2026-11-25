bash <<'OUTER'
set -euo pipefail
mkdir -p logs
source slurm/variants.sh
for T in 0 0.5 0.7; do
  for v in noUNK log; do
    variant "$v"
    export TEMP=$T MODEL_SEED=0
    sbatch --export=ALL --job-name="sel-analyze-T$T-$v" <<'EOF'
#!/bin/bash
#SBATCH --account=p_neurasearch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
source slurm/common.sh cpu
mkdir -p "$WORK/reports_T$TEMP"
python -m evaluation.analyze --corpus "$CORPUS" --rollouts "$WORK/rollouts_T$TEMP/${RUN}_[0-9]*.jsonl" --output "$WORK/reports_T$TEMP/$RUN.txt"
EOF
  done
done
OUTER
