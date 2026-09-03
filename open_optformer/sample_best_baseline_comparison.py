"""Compare, at several points along a REAL recorded optimization
trajectory, three populations' hyperparameter-suggestion behaviour:

  1. REF          -- the empirically-best real baseline algorithm for that
                      benchmark (see `find_best_baselines.py`), re-run live,
                      primed with the real recorded context.
  2. pretrained   -- the original (non-finetuned) OptFormer checkpoint,
                      conditioned on that same best algorithm's token.
  3. finetuned    -- the checkpoint finetuned on best-trajectory-only data
                      (`--rename_best`), conditioned on algorithm="best".

Context comes from the REAL recorded results.csv.zip of the winning
algorithm's own seed-runs (via `load_real_trajectory`), not a freshly
simulated trajectory -- so this replays what actually happened, truncated
to `--context_depths` trial counts (e.g. 5 = right after the standard
random-warmup phase, 40 = mid-optimization, 80 = near the end of a
100-trial budget).

Usage (run `find_best_baselines.py` first to produce --best_baselines_json):

  python -m open_optformer.sample_best_baseline_comparison \\
      --best_baselines_json best_baselines.json \\
      --results_path /path/to/raw_data_bbo_pile \\
      --context_depths 5 40 80 \\
      --pretrained_checkpoint /path/to/checkpoints/original/v0.8/qwen3_2M_.../final \\
      --pretrained_label 2M-pretrained \\
      --finetuned_checkpoint /path/to/checkpoints/only-best/qwen3_2M_.../step-00017532 \\
      --finetuned_label 2M-opt-best \\
      --out_dir out/2M

Omit --pretrained_checkpoint/--finetuned_checkpoint to skip that series
(e.g. to compute REF only once, then run per-model-size with --skip_ref).
One CSV per benchmark is written to --out_dir immediately after that
benchmark finishes (incremental save: at most one benchmark's work is lost
if the job is interrupted, and reruns skip benchmarks already done via
--skip_existing, on by default).
"""
import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from open_optformer.sample_distribution import (
    _rows,
    load_optformer_model,
    load_real_trajectory,
    sample_optformer_configs,
    sample_reference_configs,
    true_metric_range_from_trials,
)


def _augment(rows, **extra):
    for row in rows:
        row.update(extra)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--best_baselines_json", type=Path, required=True,
                    help="Output of find_best_baselines.py.")
    p.add_argument("--results_path", type=Path, required=True,
                    help="Root of the raw syne-tune results (raw_data_bbo_pile).")
    p.add_argument("--benchmark", type=str, nargs="+", default=None, dest="benchmarks",
                    help="Benchmarks to compare. Defaults to every benchmark "
                         "present in --best_baselines_json.")
    p.add_argument("--context_depths", type=int, nargs="+", default=[5, 40, 80],
                    help="Trial counts along the real trajectory to compare "
                         "at. Capped to the trajectory's actual length per "
                         "benchmark/seed (recorded as depth_effective).")
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="Which of the winning algorithm's recorded seed-runs "
                         "to use as context (also seeds the Monte Carlo "
                         "draws). A seed missing from a benchmark's "
                         "seed_experiments is skipped with a warning.")
    p.add_argument("--pretrained_checkpoint", type=Path, default=None,
                    help="Original (non-finetuned) checkpoint. Omit to skip "
                         "the pretrained-imitation series.")
    p.add_argument("--pretrained_label", type=str, default="pretrained")
    p.add_argument("--finetuned_checkpoint", type=Path, default=None,
                    help="Checkpoint finetuned on best-trajectory-only data. "
                         "Omit to skip the finetuned OPT-best series.")
    p.add_argument("--finetuned_label", type=str, default="opt-best")
    p.add_argument("--skip_ref", action="store_true",
                    help="Skip REF sampling (use when it was already "
                         "computed in a prior, checkpoint-free run).")
    p.add_argument("--oracle_metric_range", action="store_true",
                    help="Debug/ablation: quantize the pretrained/finetuned "
                         "series' proposals against the TRUE (min, max) "
                         "metric value of the held-out task's full recorded "
                         "trajectory, instead of the causal default -- tests "
                         "whether a gap to REF is a quantization artifact "
                         "rather than a model-capability issue (see "
                         "History.metric_range_override). Appends "
                         "'-oracle-range' to --pretrained_label/"
                         "--finetuned_label so runs stay distinguishable.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.2)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false",
                    default=True,
                    help="Recompute a benchmark even if its output CSV "
                         "already exists (default: skip existing, for "
                         "resumability after a job timeout).")
    args = p.parse_args()

    if args.oracle_metric_range:
        args.pretrained_label = f"{args.pretrained_label}-oracle-range"
        args.finetuned_label = f"{args.finetuned_label}-oracle-range"

    with open(args.best_baselines_json) as f:
        best_baselines = json.load(f)

    benchmarks = args.benchmarks or list(best_baselines.keys())
    print(f"Benchmarks ({len(benchmarks)}): {benchmarks}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pretrained_model = pretrained_tokenizer = None
    if args.pretrained_checkpoint is not None:
        print(f"Loading pretrained checkpoint {args.pretrained_checkpoint} ...")
        pretrained_model, pretrained_tokenizer, _ = load_optformer_model(
            args.pretrained_checkpoint,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    finetuned_model = finetuned_tokenizer = None
    if args.finetuned_checkpoint is not None:
        print(f"Loading finetuned checkpoint {args.finetuned_checkpoint} ...")
        finetuned_model, finetuned_tokenizer, _ = load_optformer_model(
            args.finetuned_checkpoint,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    try:
        max_depth = max(args.context_depths)

        for benchmark in benchmarks:
            out_csv = args.out_dir / f"{benchmark}.csv"
            if args.skip_existing and out_csv.exists():
                print(f"Skipping {benchmark}: {out_csv} already exists.")
                continue

            baseline = best_baselines.get(benchmark)
            if baseline is None:
                print(f"WARNING: no best-baseline entry for {benchmark!r} in "
                      f"{args.best_baselines_json}, skipping.")
                continue
            algorithm = baseline["algorithm"]
            seed_experiments = baseline["seed_experiments"]

            rows = []
            for seed in args.seeds:
                experiment_name = seed_experiments.get(str(seed))
                if experiment_name is None:
                    print(f"WARNING: no recorded seed={seed} run of "
                          f"{algorithm!r} for {benchmark!r}, skipping this seed.")
                    continue

                cs, metric, trials = load_real_trajectory(
                    experiment_name, args.results_path, max_depth,
                )

                metric_range = None
                if args.oracle_metric_range:
                    # Full (untruncated) trajectory, so the oracle range
                    # isn't itself capped to max_depth.
                    _, _, full_trials = load_real_trajectory(
                        experiment_name, args.results_path, None,
                    )
                    metric_range = true_metric_range_from_trials(full_trials)

                for depth in args.context_depths:
                    eff_depth = min(depth, len(trials))
                    designs = [t.config for t in trials[:eff_depth]]
                    obs = [t.metric for t in trials[:eff_depth]]
                    extra = dict(
                        real_experiment_name=experiment_name,
                        depth_requested=depth, depth_effective=eff_depth,
                    )

                    if not args.skip_ref:
                        try:
                            configs = sample_reference_configs(
                                cs, metric, algorithm, designs, obs, seed,
                                args.num_samples,
                            )
                            rows.extend(_augment(_rows(
                                configs, series="REF", benchmark=benchmark,
                                method=algorithm, algorithm=algorithm,
                                n_context_trials=eff_depth, seed=seed,
                            ), **extra))
                        except Exception as e:
                            print(f"WARNING: REF sampling failed for "
                                  f"benchmark={benchmark} algorithm={algorithm} "
                                  f"depth={depth}: {e!r}; skipping REF series "
                                  f"for this (benchmark, depth).")

                    if pretrained_model is not None:
                        configs = sample_optformer_configs(
                            args.pretrained_checkpoint, benchmark, algorithm,
                            cs, metric, designs, obs, seed, args.num_samples,
                            model=pretrained_model, tokenizer=pretrained_tokenizer,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                            metric_range=metric_range,
                        )
                        rows.extend(_augment(_rows(
                            configs, series=args.pretrained_label,
                            benchmark=benchmark, method=algorithm,
                            algorithm=algorithm, n_context_trials=eff_depth,
                            seed=seed,
                        ), **extra))

                    if finetuned_model is not None:
                        configs = sample_optformer_configs(
                            args.finetuned_checkpoint, benchmark, "best",
                            cs, metric, designs, obs, seed, args.num_samples,
                            model=finetuned_model, tokenizer=finetuned_tokenizer,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                            metric_range=metric_range,
                        )
                        rows.extend(_augment(_rows(
                            configs, series=args.finetuned_label,
                            benchmark=benchmark, method=algorithm,
                            algorithm="best", n_context_trials=eff_depth,
                            seed=seed,
                        ), **extra))

            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"wrote {out_csv} ({len(rows)} rows)")
    finally:
        del pretrained_model, pretrained_tokenizer, finetuned_model, finetuned_tokenizer
        gc.collect()
