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

## Follow-up runs (decoding, scale, prefix / y-normalization ablation)
Every model's rollouts are stored under a **label**: the dataset, plus model size / tokens / temperature where they
differ from the defaults (5M, 300M, T=1): `R`, `R_T0.5`, `R_greedy`, `R_1.2B`, `R_10M_1.2B`, `O_causal`, ...
Variables (set as a prefix on `sbatch`, they are passed into the job):

| variable | used by | meaning |
|---|---|---|
| `DATASET_ARRAY` | export, preprocess, train (array jobs) | datasets of the array tasks (default `R O`); match `--array` |
| `DATASETS` | rollouts, analyze | datasets to run (default `R O`) |
| `MODEL`, `TOKENS` | train, rollouts, score | `qwen3_5M` / `qwen3_10M`, training tokens (default 300M) |
| `ROLLOUT_TEMPERATURE` | rollouts | 1 (default), 0.5, 0 = greedy |
| `ROLLOUT_TAGS` | rollouts | prompt tag templates (default all five) |
| `SCORE_DATASETS` | score | default `R` |
| `LABELS`, `REPORT_NAME`, `PLOT_TAGS` | analyze | labels to report; with >= 2 the first is the reference of `compare/<REPORT_NAME>`; plots in `plots/<REPORT_NAME>` |

Datasets `O_none` (no prefix lines), `O_random` (2 random-length prefixes per trajectory), `O_causal` (causal y
normalization, no prefixes) each differ from `O` in one thing (`ARGS` in `gonogo.sh`); the y normalization of each
model is passed to rollouts and scoring automatically.

## Q1 (diverse vs best data), Q2 (strategy selection), scale
No oracle in the training data; random prefixes (2 per trajectory), global y normalization. The best fixed teacher
and the per-function best come from the step 0 ranking (`$OUT/ranks/teacher_ranks.parquet`, pool 4 MLE + RS,
floored AUC).

| dataset | data | default rollout tags |
|---|---|---|
| `ALL` | 4 MLE + RS, kernel_mode tags, tag dropout 0.2 | `GP-{K}-MLE`, fixed teacher's tag, `UNK` |
| `BEST_fixed` | only the best fixed teacher (train functions) | fixed teacher's tag |
| `BEST_only` | per function only the best teacher of its family (chosen on train), tag `BEST` | `BEST` |
| `ALL_BEST` | `ALL` + `BEST_only` lines (export after both; it concatenates them) | `BEST`, `GP-{K}-MLE`, fixed, `UNK` |
| `ALL_sym8` | `ALL` with all 8 square symmetries per trajectory (8x unique data) | as `ALL` |

Why per family: in the step 0 ranking, per-function selection on held-out seeds is no better than the best fixed
teacher (which teacher wins a function is mostly seed noise), while the best teacher per family is 0.035 better
(floored AUC, test; up to 0.08 on m12). The family is not in the prompt: BEST has to infer it from the observations.
Analysis: the student prompted `BEST` is compared with the selection baselines `sel_fixed` (best fixed teacher),
`sel_family` (best teacher of the function's family: the target), `sel_loso` (per-function best chosen on the other
seeds), `sel_mixture` (random teacher), `sel_hindsight` (best on the same seed). The decisive Q2 comparison is
student vs student of the same model (`ALL_BEST`: `BEST` vs the fixed teacher's tag), because the imitation gap is
larger than the selection headroom. `LABELS` entries `label@TAG` (`@FIXED` = the fixed teacher's tag) pair arms prompted differently,
e.g. `LABELS="BEST_fixed_1.2B@FIXED ALL_1.2B@FIXED ALL_1.2B@UNK BEST_only_1.2B@BEST ALL_BEST_1.2B@BEST"`.
`SKIP_EVAL=1` skips the per-label reports (only comparison + plots). `PREPROCESS_WORKERS` sets the litdata workers.
Exports above 4 GB (`ALL_sym8`) are shuffled in two passes (64 random buckets) instead of one `shuf` in memory.

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
`syne-tune`, `pandas`, `pyarrow` and `transformers`. Python 3.13 (Barnard) needs `numpy>=2` and `pyarrow>=18` (no wheels for older versions); Python 3.12 (Capella)
keeps `numpy==1.26.3` with `pyarrow<17`. `requirements.txt` selects these by Python version. Avoid pyarrow 17.0:
it ships a stray top-level `benchmarks` package that shadows the repo's.

## Resources (rough, full corpus)
export ~1.9M trajectories per dataset in minutes on 32 cores; train 5M × 300M tokens ≈ 18k steps; rollouts
400 test_core functions × 5 seeds × 5 tags × 2 models ≈ 20k runs over 8 GPU shards. Time limits in the headers are
generous; adjust after the first run.
