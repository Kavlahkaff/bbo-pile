# Optimizer selection with a BBO-Pile-style model

A small language model is trained on optimizer trajectories in 2D. Different function families have different winners:
GP-BO on GP samples, Nelder-Mead on Rosenbrock, attractive sector and different powers. An ill-conditioned ellipsoid is
held out for testing. Can the model imitate each optimizer by tag (GP, NM, RS), and pick the right one by itself (tag BEST)?
Based on [bbo-pile](https://github.com/syne-tune/bbo-pile).

| directory | contents |
|---|---|
| `datagen/` | functions, optimizers (GP = GP-UCB Matern-3/2, NM = Nelder-Mead, RS = random search), corpus generator, corpus check |
| `generate_training_data/` | export to text, tokenizer, litdata chunks |
| `open_optformer/` | bbo-pile core: prompt format (`history.py`), training (`training/`) |
| `evaluation/` | prompt, closed-loop rollouts, report, comparison, sampling match |
| `configs/` | model configs (5M, 30M, 80M), config generator |
| `slurm/` | jobs; paths in `env.sh`, runs in `variants.sh` |

## State
From the last round (archived, see below):
- `log` y encoding beats `causal_log` for NM. At temperature 0 the 5M `log` model nearly equals the NM teacher; GP needs T of 0.7 or more.
- 30M closes about 40% of the GP gap of 5M, and little of the NM gap.
- The prompt hides where the 5-10 random initial points end, so the student starts late at T = 1, whatever its size.

Now: do `log` and 30M add up, and does marking the end of the initial design help, above all for BEST?
Does 80M close more of the GP gap than 30M? UNK is not used any more.

| variant | difference |
|---|---|
| `log` | baseline; trained at 5M (seeds 0-2). Named `ALL_BEST_log_noUNK` on disk |
| `log_marker` | one `INIT` token after the last initial point; own tokenizer (`tokenizer_marker`) |

## Design
- 50 trials including 5-10 shared random initial points. All points lie on the 1000-level token grid.
- 500 test functions per family (5 seeds), 500 val, and 99,996 train functions for gp and 33,332 for each of the other
  trained families (2 seeds). Every optimizer runs on every function.
- **y encoding:** log distance to the best value in the prompt, rescaled at every step (`log_tokens`).
- **Metrics:** mean log10 normalised regret (+ 1e-8, no floor) and final win rate, next to mean linear regret.
- Temperatures were only judged on test functions so far; a final one must be chosen on others.

## Run
From the repo root (the corpus exists; `submit_cpu.sh` would regenerate it). Nothing that exists is overwritten.

    bash slurm/submit_prepare.sh                         # Barnard: export + tokenize what is missing (log_marker)
    DRY=1 bash slurm/submit_gpu.sh                       # Capella: list; without DRY: train -> rollouts
    bash slurm/submit_analyze.sh                         # Barnard: reports and compare_<time>.txt
    bash slurm/submit_match.sh student | report          # Capella | Barnard: sampling match
    bash slurm/submit_temp.sh rollouts                   # Capella: temperature sweep; then TEMP=0.7 bash slurm/submit_analyze.sh

`RUNS="log_marker qwen3_5M 2.4B 5e-3 0"` (variant model tokens lr seed, separated by `;`) selects runs in all of them.

Tests: `python -m pytest -q tst`.

## Archive
All other variants and their results are in `archive_2026-11-25/` (not in git, meant to be stored elsewhere); the code that ran them is
commit 0cf61c1.
