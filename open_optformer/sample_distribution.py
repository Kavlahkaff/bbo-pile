"""Sample HP distributions from OptFormer (and reference algorithms) and
write them to a long-format CSV.

  python -m open_optformer.sample_distribution \\
      --benchmark fcnet-protein \\
      --method RS \\
      --n_context_trials 20 \\
      --num_samples 5000 \\
      --seeds 0 1 2 3 4 \\
      --checkpoint /path/to/qwen3_2M_... \\
      --checkpoint /path/to/qwen3_5M_... \\
      --out_csv fcnet_RS_n20.csv

To sample from checkpoints that were finetuned on "best"-only trajectories
(the `--rename_best` data variant, evaluated as `OPT-best`), build the
context with a reference algorithm as usual but condition OptFormer on the
"best" algorithm token via `--algorithm`:

  python -m open_optformer.sample_distribution \\
      --benchmark fcnet-protein --benchmark nas201-ImageNet16-120 \\
      --method RS --algorithm best \\
      --checkpoint /path/to/finetuned/only-best/qwen3_2M_.../step-00017532 \\
      --checkpoint /path/to/finetuned/only-best/qwen3_80M_.../step-00000731 \\
      --out_csv opt_best_n20.csv

`--benchmark` accepts multiple values; without it, all benchmarks listed in
`generate_training_data/validation_tasks.json` are used (see
`_default_validation_benchmarks`).
"""
import argparse
import functools
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from syne_tune.backend.trial_status import Trial
from syne_tune.blackbox_repository import add_surrogate, load_blackbox
from syne_tune.experiments import ExperimentResult
from syne_tune.optimizer.baselines import BORE, CQR, RandomSearch, TPE
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)

from open_optformer.history import History
from open_optformer.optformer_searcher import OptformerScheduler, load_optformer_model


def _rea_factory(cs, metric, seed):
    # Mirrors `Methods.REA` in benchmarks/syne_tune_benchmarks/baselines.py
    # exactly, so REF sampling matches how the real "best" REA trajectories
    # in the training/eval data were actually generated.
    return SingleObjectiveScheduler(
        config_space=cs, searcher="regularized_evolution", metric=metric,
        do_minimize=True, random_seed=seed,
        searcher_kwargs={"population_size": 20, "sample_size": 10},
    )


def _hebo_factory(cs, metric, seed):
    # HEBO's searcher is disabled repo-wide (missing dependency) as of this
    # writing -- import lazily so this only fails at call time, with a
    # clear ImportError, rather than breaking every import of this module.
    # Once HEBO is working on the cluster, this starts working with no
    # further code change.
    from open_optformer.hebo_searcher import HEBOSearcher

    return SingleObjectiveScheduler(
        config_space=cs,
        searcher=HEBOSearcher(
            config_space=cs, do_minimize=True, random_seed=seed,
        ),
        metric=metric, do_minimize=True, random_seed=seed,
    )


_REF_FACTORIES = {
    "RS":   lambda cs, metric, seed: RandomSearch(
        config_space=cs, metrics=[metric], do_minimize=True, random_seed=seed,
    ),
    "CQR":  lambda cs, metric, seed: CQR(
        config_space=cs, metric=metric, do_minimize=True, random_seed=seed,
    ),
    "TPE":  lambda cs, metric, seed: TPE(
        config_space=cs, metric=metric, do_minimize=True, random_seed=seed,
    ),
    "BORE": lambda cs, metric, seed: BORE(
        config_space=cs, metric=metric, do_minimize=True, random_seed=seed,
    ),
    "REA":  _rea_factory,
    "HEBO": _hebo_factory,
}

# "best" isn't a runnable reference scheduler -- it only exists as an
# OptFormer conditioning token, produced by `compile_data.py --rename_best`
# when training on best-trajectory-only data (the `OPT-best` method in
# `benchmarks/*/baselines.py`). It's valid as `--algorithm` but not as
# `--method` (context trajectories still need an actual scheduler to drive
# them).
ALGORITHMS = list(_REF_FACTORIES) + ["best"]

# Families not covered by `--benchmark`'s default (see
# `_default_validation_benchmarks`): global-optimization problems use a
# live-function `AskTellScheduler` interface (`benchmarks/
# global_optimization_problems/run_benchmark.py`), not the
# `load_blackbox`/`add_surrogate` interface `_load_blackbox` relies on.
_UNSUPPORTED_VALIDATION_FAMILIES = ("global_optimization_problems",)


@functools.lru_cache(maxsize=None)
def _load_blackbox(benchmark_name: str):
    """Load the blackbox + surrogate for a benchmark (cached: the same
    benchmark is loaded once even if reused across seeds/checkpoints)."""
    from benchmarks.syne_tune_benchmarks.fcnet_benchmarks import (
        fcnet_benchmark_definitions,
    )
    from benchmarks.syne_tune_benchmarks.hpob_benchmarks import (
        hpob_benchmark_definitions,
    )
    from benchmarks.syne_tune_benchmarks.lcbench_benchmarks import (
        lcbench_benchmark_definitions,
    )
    from benchmarks.syne_tune_benchmarks.nas201_benchmarks import (
        nas201_benchmark_definitions,
    )
    from benchmarks.syne_tune_benchmarks.pd1_benchmarks import (
        pd1_benchmark_definitions,
    )
    from benchmarks.syne_tune_benchmarks.tabrepo_benchmarks import (
        tabrepo_benchmark_definitions,
    )

    defs = {
        **fcnet_benchmark_definitions,
        **lcbench_benchmark_definitions,
        **nas201_benchmark_definitions,
        **tabrepo_benchmark_definitions,
        **pd1_benchmark_definitions,
        **hpob_benchmark_definitions,
    }
    bd = defs[benchmark_name]
    bb = load_blackbox(bd.blackbox_name)[bd.dataset_name]
    bb = add_surrogate(blackbox=bb, predict_curves=False)
    return bb, bd


def _default_validation_benchmarks(
    validation_tasks_json: Path,
    exclude_families: Tuple[str, ...] = _UNSUPPORTED_VALIDATION_FAMILIES,
) -> List[str]:
    """Flatten `generate_training_data/validation_tasks.json` (the held-out
    benchmark list `compile_data.py` uses to mark validation trajectories)
    into a single benchmark-name list, skipping families `_load_blackbox`
    can't load."""
    import json

    with open(validation_tasks_json) as f:
        families = json.load(f)

    benchmarks = []
    skipped = []
    for family, names in families.items():
        if family in exclude_families:
            skipped.append(family)
            continue
        benchmarks.extend(names)

    if skipped:
        print(
            f"Skipping validation-task families not supported by "
            f"_load_blackbox: {skipped} (pass --benchmark explicitly to "
            f"include them via a different evaluation path)."
        )
    return benchmarks


def _draw_method_trace(bb, cs, metric, seed, n_trials, mode_sign, method,
                       n_warmup: int = 5):
    """Generate a context trajectory by running `method` for n_trials steps.

    First n_warmup steps are uniform-random; the remaining (n_trials - n_warmup)
    steps come from `method`'s suggest(). All steps are evaluated on the
    blackbox and fed back to the scheduler.

    Returns (configs, signed_ys).
    """
    fidelity_key = list(bb.fidelity_space.keys())[0]
    max_fidelity = max(bb.fidelity_values)
    rng = np.random.RandomState(seed)
    sched = _REF_FACTORIES[method](cs, metric, seed)
    configs, ys = [], []
    for i in range(n_trials):
        if i < n_warmup:
            cfg = {
                k: v.sample(random_state=rng) if hasattr(v, "sample") else v
                for k, v in cs.items()
            }
        else:
            cfg = sched.suggest().config
        y_signed = mode_sign * float(bb(cfg, fidelity={fidelity_key: max_fidelity})[metric])
        sched.on_trial_complete(Trial(i, cfg, 0.0), {metric: y_signed})
        configs.append(cfg)
        ys.append(y_signed)
    return configs, ys


def build_context(
    benchmark_name: str, method: str, seed: int, n_context_trials: int,
    n_warmup: int = 5,
):
    """Build a method-driven context trajectory once. Returns (cs, metric,
    designs, signed_obs); pass these to sample_reference_configs / sample_optformer_configs
    to ensure REF and OptFormer see the IDENTICAL context."""
    if method not in _REF_FACTORIES:
        raise ValueError(f"Unknown method {method!r}.")
    bb, bd = _load_blackbox(benchmark_name)
    cs = bb.configuration_space
    metric = bd.metric
    mode_sign = -1.0 if bd.mode == "max" else 1.0
    designs, obs = _draw_method_trace(
        bb, cs, metric, seed, n_context_trials, mode_sign, method, n_warmup,
    )
    return cs, metric, designs, obs


def load_real_trajectory(experiment_name: str, results_path: Path, max_trials: int):
    """Load the REAL recorded trajectory of one experiment (not a
    re-simulated one), truncated to its first `max_trials` trials.

    Reuses `generate_training_data/load_data.py`'s own loading path
    (`get_config_space_from_metadata` + `load_result` +
    `History.from_syne_tune_experiment`) so this is byte-for-byte the same
    data/config-space source the pretraining/finetuning pipeline itself
    used to build `History` prompts for this experiment.

    Returns (cs, metric_name, trials), where `trials` is a list of
    `Trial(config, metric)` in chronological order, with `metric` already
    sign-flipped to always-minimize (matching the convention
    `sample_reference_configs`/`sample_optformer_configs` expect -- see
    `build_context`'s `mode_sign`).
    """
    # generate_training_data/ isn't a package (its own files import each
    # other with flat `from load_data import ...`); add it to sys.path
    # lazily so this also works when this module is imported from outside
    # that directory (as intended: `open_optformer/sample_*` scripts run
    # with the repo root on PYTHONPATH, not generate_training_data/).
    import sys

    gen_data_dir = str(Path(__file__).resolve().parent.parent / "generate_training_data")
    if gen_data_dir not in sys.path:
        sys.path.insert(0, gen_data_dir)
    from load_data import get_config_space_from_metadata, load_result, read_single_metadata

    metadata_path = results_path / experiment_name / "metadata.json"
    _, metadata = read_single_metadata((str(metadata_path), str(results_path)))
    if metadata is None:
        raise FileNotFoundError(f"Could not read metadata for {experiment_name!r} under {results_path}")

    config_space = get_config_space_from_metadata(metadata)
    metric_name = metadata["metric_names"][0]
    res = load_result(experiment_name, metric_name, config_space, results_path)
    if res is None:
        raise FileNotFoundError(f"Could not read results.csv.zip for {experiment_name!r} under {results_path}")

    hist = History.from_syne_tune_experiment(
        ExperimentResult(name=experiment_name, metadata=metadata, results=res,
                          path=results_path, tuner=None),
        max_num_trials=max_trials,
    )
    return config_space, metric_name, hist.trials


def sample_reference_configs(
    cs, metric: str, method: str, designs, obs, seed: int,
    num_samples: int = 5000,
) -> List[Dict[str, Any]]:
    """Draw `num_samples` configs from the reference algorithm `method`,
    each from a fresh scheduler primed with the same (designs, obs) context."""
    if method not in _REF_FACTORIES:
        raise ValueError(f"Unknown method {method!r}.")
    out = []
    for j in range(num_samples):
        sched = _REF_FACTORIES[method](cs, metric, seed * num_samples + j)
        for i, (cfg, y) in enumerate(zip(designs, obs)):
            sched.on_trial_complete(Trial(i, cfg, 0.0), {metric: y})
        out.append(sched.suggest().config)
    return out


def sample_optformer_configs(
    checkpoint_dir, benchmark_name: str, algorithm: str,
    cs, metric: str, designs, obs, seed: int, num_samples: int = 5000,
    model=None, tokenizer=None, gpu_memory_utilization: float = 0.2,
    metric_range: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """Sample `num_samples` configs from OptFormer conditioned on (designs, obs).

    `algorithm` sets the conditioning token (`task_info["algorithm"]`) seen
    by the model -- e.g. "best" for checkpoints finetuned on best-trajectory
    data (`OPT-best`). It is independent of how (designs, obs) itself was
    generated (see `build_context`'s `method` argument): OptFormer is asked
    "what would `algorithm` suggest next", given a history that may have
    been produced by a different reference scheduler.

    Pass a preloaded `model`/`tokenizer` (from `load_optformer_model`) to
    reuse them across seeds/benchmarks instead of reloading vLLM each call.

    `metric_range`: debug/ablation knob (see `History.metric_range_override`)
    -- pass the task's TRUE (y_min, y_max) here (e.g. from
    `true_metric_range_from_trials`) to quantize against that fixed oracle
    range instead of the causal default, to test whether a gap to baselines
    is a quantization artifact.
    """
    scheduler = OptformerScheduler(
        config_space=cs, metric=metric, checkpoint_dir=Path(checkpoint_dir),
        task_info={"name": benchmark_name, "algorithm": algorithm, "metric_names": metric},
        do_minimize=True, random_seed=seed, n_sample_configurations=num_samples,
        model=model, tokenizer=tokenizer,
        gpu_memory_utilization=gpu_memory_utilization,
        metric_range=metric_range,
    )
    for i, (cfg, y) in enumerate(zip(designs, obs)):
        scheduler.on_trial_complete(Trial(i, cfg, 0.0), {metric: y})
    configs, _ = scheduler.searcher._sample_n_configs()
    return configs


def true_metric_range_from_trials(
    trials, percentile_lo: float = 0.0, percentile_hi: float = 95.0,
) -> Tuple[float, float]:
    """The task's TRUE (oracle) (min, max) metric value, from a list of
    `Trial`s (already sign-flipped to always-minimize) covering the task's
    FULL recorded history -- pass `max_trials=None` to `load_real_trajectory`
    to get the untruncated trial list this expects, not one truncated to a
    context depth. Debug/ablation use: see `sample_optformer_configs`'s
    `metric_range` argument.

    Percentile-clipped by default (same (0, 95) defaults as
    `History.metric_percentile_lo/_hi`'s causal range), NOT the raw min/max:
    a single diverged trial in the full trajectory would otherwise consume
    most of the oracle range's quantization resolution, which would make an
    "oracle vs. causal" comparison confound range *source* (true vs.
    observed-so-far) with range *robustness* (percentile-clipped vs. not).
    Pass percentile_lo=0, percentile_hi=100 for the raw extent instead.
    """
    metrics = [t.metric for t in trials]
    y_min = np.percentile(metrics, percentile_lo)
    y_max = np.percentile(metrics, percentile_hi)
    if y_min == y_max:
        y_max += 1  # avoid division by zero in quantization, matches History
    return (float(y_min), float(y_max))


def _series_label(checkpoint_dir: Path, taken: set) -> str:
    """Derive a series label distinguishing checkpoints from different
    finetuning runs.

    `--checkpoint` may point directly at a run directory (`ckpt.name` is
    already the run name, e.g. a pretrain run's `final/`'s parent) or --
    for finetuned checkpoints converted by `convert_checkpoints.sh` -- at a
    `<run_name>/<step-NNNNNNN or final>` leaf, where `ckpt.name` alone
    ("step-00017532") does not distinguish one model size/config from
    another. Fall back to the parent directory name in that case.
    """
    name = checkpoint_dir.name
    if name == "final" or name.startswith("step-"):
        label = f"{checkpoint_dir.parent.name}/{name}"
    else:
        label = name

    if label in taken:
        # Disambiguate by keeping more of the path (rare: only if two
        # checkpoints truly share the derived label).
        label = str(checkpoint_dir)
    taken.add(label)
    return label


def _rows(
    configs, *, series, benchmark, method, n_context_trials, seed,
    algorithm=None,
):
    """Convert a list of sampled configs into long-format CSV rows.

    `method` is the scheduler that generated the context trajectory;
    `algorithm` (only set for OptFormer rows) is the conditioning token
    used to sample -- they differ e.g. for OPT-best sampling, where the
    context is RS-driven but `algorithm="best"`.
    """
    return [
        {
            "series": series,
            "benchmark": benchmark,
            "method": method,
            "algorithm": algorithm if algorithm is not None else method,
            "n_context_trials": n_context_trials,
            "seed": seed,
            "sample_idx": i,
            **{f"hp_{k}": v for k, v in cfg.items()},
        }
        for i, cfg in enumerate(configs)
    ]


if __name__ == "__main__":
    default_validation_json = (
        Path(__file__).resolve().parent.parent
        / "generate_training_data" / "validation_tasks.json"
    )

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, action="append", required=True,
                   dest="checkpoints",
                   help="HF checkpoint directory. Pass multiple times, e.g. "
                        "once per finetuned model size.")
    p.add_argument("--checkpoint_label", type=str, action="append", default=None,
                   dest="checkpoint_labels",
                   help="Series label for each --checkpoint, in the same "
                        "order (e.g. '2M', '5M', ...). Pass once per "
                        "--checkpoint, or omit to auto-derive from the "
                        "checkpoint path (see `_series_label`).")
    p.add_argument("--benchmark", type=str, nargs="+", default=None,
                   dest="benchmarks",
                   help="Benchmark name(s) to sample on. Defaults to every "
                        "benchmark listed in --validation_tasks_json.")
    p.add_argument("--validation_tasks_json", type=Path,
                   default=default_validation_json,
                   help="Used only when --benchmark is omitted.")
    p.add_argument("--method", type=str, required=True, choices=list(_REF_FACTORIES),
                   help="Reference scheduler used to drive the context "
                        "trajectory (and, unless --skip_reference, to "
                        "produce the REF series).")
    p.add_argument("--algorithm", type=str, default=None, choices=ALGORITHMS,
                   help="OptFormer conditioning token (task_info['algorithm']). "
                        "Defaults to --method. Use 'best' for checkpoints "
                        "finetuned on best-trajectory data (OPT-best).")
    p.add_argument("--skip_reference", action="store_true",
                   help="Skip sampling the REF series (only sample from "
                        "--checkpoints). Saves time when only the "
                        "checkpoint distributions are needed.")
    p.add_argument("--n_context_trials", type=int, default=10,
                   help="Number of context trials (warmup + method-driven).")
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--gpu_memory_utilization", type=float, default=0.2)
    p.add_argument("--out_csv", type=Path, required=True)
    args = p.parse_args()

    algorithm = args.algorithm or args.method
    benchmarks = args.benchmarks or _default_validation_benchmarks(
        args.validation_tasks_json,
    )
    print(f"Benchmarks ({len(benchmarks)}): {benchmarks}")

    taken_labels = set()
    checkpoint_labels = args.checkpoint_labels
    if checkpoint_labels is not None:
        assert len(checkpoint_labels) == len(args.checkpoints), (
            f"Got {len(checkpoint_labels)} --checkpoint_label but "
            f"{len(args.checkpoints)} --checkpoint."
        )
    else:
        checkpoint_labels = [
            _series_label(ckpt, taken_labels) for ckpt in args.checkpoints
        ]

    rows = []

    # Context trajectories (and, unless skipped, REF samples) only depend
    # on --method/--benchmark/--seeds, not on the checkpoints -- build them
    # once and reuse for every checkpoint below.
    contexts = {}
    for benchmark in benchmarks:
        for seed in args.seeds:
            print(f"Building {args.method} context: benchmark={benchmark} seed={seed} ...")
            contexts[(benchmark, seed)] = build_context(
                benchmark, args.method, seed, args.n_context_trials,
            )

    if not args.skip_reference:
        for benchmark in benchmarks:
            for seed in args.seeds:
                cs, metric, designs, obs = contexts[(benchmark, seed)]
                print(f"Sampling {args.method} reference: benchmark={benchmark} seed={seed} ...")
                rows.extend(_rows(
                    sample_reference_configs(
                        cs, metric, args.method, designs, obs, seed, args.num_samples,
                    ),
                    series="REF", benchmark=benchmark, method=args.method,
                    n_context_trials=args.n_context_trials, seed=seed,
                ))

    # One vLLM load per checkpoint, reused across every benchmark/seed
    # combination for that checkpoint (mirrors the shared-model-loading
    # pattern in benchmarks/syne_tune_benchmarks/benchmark_main.py).
    for ckpt, label in zip(args.checkpoints, checkpoint_labels):
        print(f"Loading checkpoint {label} from {ckpt} ...")
        model, tokenizer, _ = load_optformer_model(
            ckpt, gpu_memory_utilization=args.gpu_memory_utilization,
        )
        try:
            for benchmark in benchmarks:
                for seed in args.seeds:
                    cs, metric, designs, obs = contexts[(benchmark, seed)]
                    print(f"Sampling OptFormer {label} (algorithm={algorithm}): "
                          f"benchmark={benchmark} seed={seed} ...")
                    rows.extend(_rows(
                        sample_optformer_configs(
                            ckpt, benchmark, algorithm,
                            cs, metric, designs, obs, seed, args.num_samples,
                            model=model, tokenizer=tokenizer,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                        ),
                        series=label, benchmark=benchmark, method=args.method,
                        algorithm=algorithm,
                        n_context_trials=args.n_context_trials, seed=seed,
                    ))
        finally:
            del model, tokenizer
            gc.collect()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(rows)} rows)")
