# GP-BO trajectory corpus → bbo-pile

Student models trained on optimizer traces on 2D GP-sample functions (corpus: `master-thesisv2/gp_traj.py --preset
student`). See `HANDOFF.md` for the research questions; this file lists the code and the commands.

## Code
| step | script | output |
|---|---|---|
| 0 teacher ranking | `generate_training_data/rank_gp_teachers.py` | `teacher_ranks.parquet`, report (win rates, selection headroom, boundary loss rates, quantization floor) |
| 1 export | `generate_training_data/compile_gp_data.py` | `train.txt`, `valid.txt`, `manifest.json` (test functions are never exported) |
| 2 tokenizer | `generate_training_data/{make_tokenizer_sample,train_tokenizer,check_gp_tokenizer}.py`, `gp_tokenizer_symbols.txt` | `tokenizer/gp_v1` (vocab 1062) |
| 3 tokenize | `generate_training_data/preprocess_data.py --block_size 1025` | litdata chunks |
| 4 configs | `configs/generate_configs.py` (`qwen3_5M`, `qwen3_10M`), train with `open_optformer/training/pretrain.py` | checkpoints |
| 5 rollouts | `benchmarks/gp_functions/run_gp_benchmark.py` (`GPGridFunction` in `gp_function.py`) | `rollouts.jsonl` |
| 6 teacher-forced | `benchmarks/gp_functions/score_teacher_traces.py` | per-step NLL / greedy next x per (trace, tag) |
| 7 analysis | `benchmarks/gp_functions/analyze_gp_eval.py` | report, CSV tables, `verdict.json` |

Shared helpers (corpus reading, 8 square symmetries, hp quantization, bilinear interpolation): `open_optformer/gp_corpus.py`.

Prompt format (search space first, `History(space_first=True)`; HPO prompts are unchanged):

    search-space:{type:UNI,min_value:0.0,max_value:1.0,linear_scale}{...},algorithm:GP-RBF-ORACLE,history:470,969*708|...

340 tokens per 50-trial trajectory (351 with the hp suffix).

## Conventions
- `X[:, 0]` is the grid row, so `argmin[0]` is x0 (checked against `gp_traj.Fn`).
- Symmetries: index 0 identity, 1/2 flip x0/x1, 3 rot180, 4 transpose, 5/6 rot90/rot270, 7 anti-transpose.
  `--symmetry_set noswap` = {0,1,2,3} (S4 of A1: no axis swap, lengthscales unchanged); `d4` = all 8.
- `--num_permutation` (hyperparameter-order shuffle) is an axis swap for two identical UNI dims, so it defaults to 0
  and is rejected with `--symmetries > 1` or `--hp_suffix`.
- Tag schemes: `generic`, `kernel`, `kernel_mode` (default), `quality` (RS stays `RS`), `best`; `--hp_suffix` adds
  the true hyperparameters to any scheme; `--tag_dropout` replaces the whole tag by `UNK`.
- Prompt-tag templates in rollouts / scoring: `{K}` = function family, e.g. `GP-{K}-ORACLE` = matched oracle tag.

## Early go/no-go check
On the cluster use the SLURM jobs in `slurm/gp/` (see `slurm/gp/README.md`); they wrap the stages below.
`benchmarks/gp_functions/gonogo.sh <stage>` with stages
`rank export tokenizer preprocess configs` → train `R` and `O` (5M, 300M tokens) → `rollouts score analyze`.
Passes if, on test_core (floored metric), the oracle-prompted student is not worse than the matched MLE teacher (better =
CI below 0), and the tag-likelihood matrix of model R puts each teacher's minimum NLL in its own tag group (oracle /
MLE / RS). See `$OUT/eval/{R,O}/report.txt` and `verdict.json`.

## Datasets (each changes one factor relative to R)
| name | exporter arguments |
|---|---|
| R | `--tag_scheme kernel_mode --teachers all --tag_dropout 0.2 --sample_shorter_trajectories` (S1, all functions) |
| O | R with `--teachers oracle` (= B with oracle: no selection rule beats the oracle out of sample) |
| B_np | R with `--only_best nopriv --best_by family --ranks teacher_ranks.parquet` (best fixed MLE teacher per family, chosen on train functions) |
| B_noisy | R with `--only_best all --best_by auc --ranks ...` (per-function best, contrast only) |
| M | R with `--teachers mle` |
| R_nopriv | R with `--teachers mle,rs` |
| tag schemes | R with `--tag_scheme generic / kernel / quality`, `--hp_suffix` (with `kernel` and with `kernel_mode`) |
| A1 | R with `--symmetries 8` (S8) / `--symmetries 4 --symmetry_set noswap` (S4); scarce regime with `--frac_functions 0.125 / 0.25` |
| A2 | R with `--boundary_margin m`; control: add `--boundary_control` |
| E4 | R with `--frac_functions 0.25 / 0.5` |

Token budget and LR: sweep the 5M model on R at {300M, 600M, 1.2B, 2.4B} × {5e-3, 1e-2}, pick the knee, then keep
budget and LR fixed for all arms.
