# Optimizer selection with a BBO-Pile-style model

A small language model is trained on optimizer trajectories in 2D. Two function families have different winners:
GP samples (GP-BO wins) and Rosenbrock (Nelder-Mead wins). Can the model imitate each optimizer by tag, and pick the
right one by itself (tag BEST or UNK)? Based on [bbo-pile](https://github.com/syne-tune/bbo-pile).

| directory | contents |
|---|---|
| `datagen/` | functions, optimizers (GP = GP-UCB Matern-3/2, NM = Nelder-Mead, RS = random search), corpus generator, corpus check |
| `generate_training_data/` | export to text, tokenizer, litdata chunks |
| `open_optformer/` | bbo-pile core: prompt format (`history.py`), training (`training/`) |
| `evaluation/` | prompt, closed-loop rollouts, report |
| `configs/` | model configs, config generator |
| `slurm/` | jobs; all settings in `env.sh` |

## Design
- 50 trials including 5-10 shared random initial points. All points lie on the 1000-level token grid.
- Per family: 500 test (5 seeds), 500 val, 50,000 train functions (2 seeds). Every optimizer runs on every function.
- **y encoding** (`--y_norm causal_log`): signed log distance to the incumbent so far. It never changes once written, so a
  training prefix is exactly an inference prompt, and every improvement is visible. `log` (rescaled each step) is the
  alternative; use it with `N_PREFIXES=2`. Export and rollouts must use the same value.
- **Metrics:** mean log10 normalised regret (+ 1e-8, no floor) and final win rate, next to mean linear regret. A -3 floor is
  reported for comparison only.
- Random search is a tagged teacher and a baseline, not a winner class.

## Run
Adjust `slurm/env.sh`, then from the repo root:

    bash slurm/submit_cpu.sh       # Barnard: generate -> check (contrast must hold) -> prepare
    bash slurm/submit_gpu.sh       # Capella, afterwards: train -> rollouts
    sbatch slurm/analyze.sbatch    # Barnard: $WORK/reports/<run>.txt

Variants are set in the environment, e.g. `Y_NORM=log N_PREFIXES=2 bash slurm/submit_cpu.sh` (a finished corpus is
resumed, not recomputed) and then the same variables for the GPU and analysis jobs.

Tests: `python -m pytest -q tst`.
