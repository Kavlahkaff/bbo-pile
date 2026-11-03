# SLURM jobs for the GP go/no-go check

Every stage of `benchmarks/gp_functions/gonogo.sh` as a batch job, so nothing heavy runs on a login node.
Barnard (CPU) and Capella (GPU) have separate schedulers, so there is one submit script per cluster.
Submit **from the repo root**: the jobs use `$SLURM_SUBMIT_DIR` as the repo.

| job | cluster | what | needs |
|---|---|---|---|
| `rank.sbatch` | Barnard | step 0 teacher ranking → `$OUT/ranks` | corpus |
| `export.sbatch` (array R, O) | Barnard | exports, shuffled → `$OUT/export/{R,O}` | corpus |
| `tokenizer.sbatch` | Barnard | tokenizer gp_v1 → `$BASE/tokenizer/gp_v1`, checks → `$OUT/tokenizer_check.json` | corpus |
| `preprocess.sbatch` (array R, O) | Barnard | litdata chunks → `$BASE/tokenized_dataset/gp_v1/{R,O}` | export, tokenizer |
| `train.sbatch` (array R, O) | Capella | config + 5M training → `$BASE/checkpoints/gp_v1/gonogo_*` | preprocess |
| `rollouts.sbatch` (array, 8 shards) | Capella | student rollouts → `$OUT/rollouts` | train |
| `score.sbatch` | Capella | teacher-forced tag likelihoods → `$OUT/scores_R.parquet` | train |
| `analyze.sbatch` | Barnard | report + verdict → `$OUT/eval/{R,O}` | rollouts, score |

## Run
```bash
# 1. Barnard login node, repo root
bash slurm/gp/submit_barnard.sh          # rank, export, tokenizer, then preprocess (afterok)
# 2. Capella login node, repo root, once preprocess finished
bash slurm/gp/submit_capella.sh          # train, then rollouts + score (afterok)
# 3. Barnard login node, once rollouts and score finished
mkdir -p logs/gp && sbatch slurm/gp/analyze.sbatch
cat logs/gp/gp-analyze-<jobid>.out       # ends with both verdict.json
```
Single jobs can be (re)submitted on their own, e.g. `sbatch --array=1 slurm/gp/train.sbatch` (O only).
Rollouts resume: resubmitting skips finished runs. Logs: `logs/gp/<job>-<jobid>[_<task>].out`.

## Configure
All paths live in `slurm/gp/env.sh` (`CORPUS`, `BASE`, `OUT`, modules, venvs, `TOKENS`, `LR`, wandb key file).
`BASE` must be reachable from both clusters. Use another file without editing it:
`sbatch --export=ALL,GP_ENV_FILE=/path/my_env.sh slurm/gp/train.sbatch`.
Without `WANDB_API_KEY` (in `.env.wandb`, `KEY=VALUE` lines) training logs to csv in the checkpoint dir.

## Virtual environments (once, inside the repo)
CPU venv on **Barnard** (`.venv-datagen`):
```bash
module load release/2026 GCCcore/14.2.0 Python/3.13.1
python -m venv .venv-datagen && source .venv-datagen/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt syne-tune pytest
python -m pytest -q tst/test_gp_corpus.py tst/test_gp_export.py tst/test_history.py
```
GPU venv on **Capella** (`.venv-litgpt`):
```bash
module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0
python -m venv .venv-litgpt && source .venv-litgpt/bin/activate
pip install torch                       # CUDA build
pip install -r requirements.txt syne-tune
```
Existing venvs work too (set `VENV_CPU` / `VENV_GPU` in `env.sh`) if they have `litgpt`, `litdata`, `sentencepiece`,
`syne-tune`, `pandas`, `pyarrow` and `transformers`. With numpy < 2 use `pyarrow<17`: pyarrow 17.0 ships a stray
top-level `benchmarks` package that shadows the repo's.

## Resources (rough, full corpus)
export ~1.9M trajectories per dataset in minutes on 32 cores; train 5M × 300M tokens ≈ 18k steps; rollouts
400 test_core functions × 5 seeds × 5 tags × 2 models ≈ 20k runs over 8 GPU shards. Time limits in the headers are
generous; adjust after the first run.
