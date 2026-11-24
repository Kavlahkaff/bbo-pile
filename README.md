# Optimizer selection with a BBO-Pile-style model

A small language model is trained on optimizer trajectories in 2D. Different function families have different winners:
GP-BO on GP samples, Nelder-Mead on Rosenbrock, attractive sector and different powers. An ill-conditioned ellipsoid is
held out for testing. Can the model imitate each optimizer by tag, and pick the
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
- 500 test functions per family (5 seeds), 500 val, and 99,996 train functions for gp and 33,332 for each of the other
  trained families (2 seeds). Every optimizer runs on every function.
- **y encoding** (`--y_norm causal_log`): signed log distance to the incumbent so far. It never changes once written, so a
  training prefix is exactly an inference prompt, and every improvement is visible. `log` (rescaled each step) is the
  alternative; use it with `N_PREFIXES=2`. Export and rollouts must use the same value.
- **Metrics:** mean log10 normalised regret (+ 1e-8, no floor) and final win rate, next to mean linear regret. A -3 floor is
  reported for comparison only.
- Random search is a tagged teacher and a baseline, not a winner class.

## Run
Adjust `slurm/env.sh`, then from the repo root:

    bash slurm/submit_pilot.sh     # Barnard: do the winners hold? verdict in $PILOT/report.txt
    bash slurm/submit_cpu.sh       # Barnard, after a GO: generate -> check (contrast must hold) -> prepare all variants
    bash slurm/submit_gpu.sh       # Capella, afterwards: train -> rollouts
    bash slurm/submit_analyze.sh   # Barnard: $WORK/reports/<run>.txt

The runs are defined in `slurm/variants.sh`: `base` (causal_log), `noRS` (no random search lines), `log` and `log_p2`
(rescaled log without / with two prefixes per line), `noUNK`, `noBEST`, `noUNK_noBEST` (without these tags). Use e.g. `VARIANTS="noRS log" bash slurm/submit_gpu.sh` for a
subset and `submit_prepare.sh` to prepare a new variant of a finished corpus.

Tests: `python -m pytest -q tst`.
