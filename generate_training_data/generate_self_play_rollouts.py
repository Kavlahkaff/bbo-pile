"""Approach C (expert iteration / STaR) -- step 1: generate self-play rollouts.

Runs the current checkpoint as an actual optimizer, driven by
`OptformerScheduler` exactly as `example_searcher.py` does, for every
(training task, seed) pair. The critical correctness point (see the plan):
every proposed config is scored with the REAL kNN surrogate via
`open_optformer.sample_distribution._load_blackbox`, never with any value
the model itself might claim -- `OptFormerSearcher.suggest()` only proposes
`x`, it does not "know" `y` until we query the surrogate and feed it back
via `on_trial_complete`.

Each rollout's full (config, y) trajectory is serialized as JSON so
`select_self_play_rollouts.py` can score and filter it in a separate step
without needing the model loaded again.

Usage:
    python generate_self_play_rollouts.py \\
        --checkpoint_dir /path/to/hf/checkpoint \\
        --all_training_tasks_json all_training_tasks.json \\
        --n_trials 100 --n_seeds 5 \\
        --out_dir /path/to/self_play/round_1/qwen3_2M/rollouts
"""
import argparse
import itertools
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_EXCLUDE_FAMILIES = ("global_optimization_problems",)


def _default_train_benchmarks(all_training_tasks_json: Path,
                               exclude_families=_EXCLUDE_FAMILIES) -> list[str]:
    """Flatten all_training_tasks.json into a single benchmark-name list,
    skipping families not backed by the load_blackbox/add_surrogate
    interface `_load_blackbox` relies on (mirrors
    sample_distribution.py::_default_validation_benchmarks)."""
    with open(all_training_tasks_json) as f:
        families = json.load(f)

    benchmarks = []
    skipped = []
    for family, names in families.items():
        if family in exclude_families:
            skipped.append(family)
            continue
        benchmarks.extend(names)

    if skipped:
        logger.info(f"Skipping families not supported by _load_blackbox: {skipped}")
    return benchmarks


def _run_one_rollout(benchmark_name: str, seed: int, checkpoint_dir: Path,
                      n_trials: int, use_vllm: bool,
                      gpu_memory_utilization: float, model=None, tokenizer=None):
    """Run a single T-step self-play rollout on `benchmark_name`, scoring
    every proposed config with the real kNN surrogate. Returns a plain dict
    (JSON-serializable) of the resulting trajectory, not a History -- History
    reconstruction happens in select_self_play_rollouts.py so this script
    doesn't need open_optformer.history's syne_tune-derived config-space
    machinery just to serialize.
    """
    from open_optformer.sample_distribution import _load_blackbox
    from open_optformer.optformer_searcher import OptformerScheduler
    from syne_tune.backend.trial_status import Trial

    bb, bd = _load_blackbox(benchmark_name)
    config_space = bb.configuration_space
    metric = bd.metric
    mode_sign = -1.0 if bd.mode == "max" else 1.0
    fidelity_key = list(bb.fidelity_space.keys())[0]
    max_fidelity = max(bb.fidelity_values)

    scheduler = OptformerScheduler(
        config_space=config_space,
        metric=metric,
        checkpoint_dir=checkpoint_dir,
        task_info={"name": benchmark_name, "algorithm": "self-play", "metric_names": metric},
        random_seed=seed,
        n_sample_configurations=1,
        use_vllm=use_vllm,
        gpu_memory_utilization=gpu_memory_utilization,
        model=model,
        tokenizer=tokenizer,
    )

    trials = []
    for trial_id in range(n_trials):
        config = scheduler.suggest().config
        # The REAL surrogate, not the model's own claimed y -- this is what
        # makes this expert iteration/STaR rather than self-distillation.
        y_signed = mode_sign * float(bb(config, fidelity={fidelity_key: max_fidelity})[metric])
        scheduler.on_trial_complete(
            Trial(trial_id=trial_id, config=config, creation_time=0.0),
            {metric: y_signed},
        )
        trials.append({"config": config, "y": y_signed})

    return {
        "benchmark": benchmark_name,
        "seed": seed,
        "metric": metric,
        "mode": bd.mode,
        "trials": trials,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint_dir", type=Path, required=True,
                    help="HF-format checkpoint (required for use_vllm=True; see "
                         "open_optformer.optformer_searcher.detect_hf_checkpoint).")
    p.add_argument("--all_training_tasks_json", type=Path,
                    default=Path(__file__).resolve().parent / "all_training_tasks.json")
    p.add_argument("--benchmark", type=str, nargs="+", default=None, dest="benchmarks",
                    help="Explicit benchmark list, overriding --all_training_tasks_json "
                         "(useful for the round-0 tiny smoke test).")
    p.add_argument("--n_trials", type=int, default=100)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Explicit seed list, overriding --n_seeds (0..n_seeds-1).")
    p.add_argument("--use_vllm", action="store_true", default=True)
    p.add_argument("--no_vllm", dest="use_vllm", action="store_false")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s")

    benchmarks = args.benchmarks or _default_train_benchmarks(args.all_training_tasks_json)
    seeds = args.seeds or list(range(args.n_seeds))
    logger.info(f"{len(benchmarks)} benchmarks x {len(seeds)} seeds = "
                f"{len(benchmarks) * len(seeds)} rollouts, n_trials={args.n_trials}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load the model/tokenizer once and reuse across every rollout instead of
    # reloading vLLM per (benchmark, seed) -- same pattern as
    # sample_distribution.py's load_optformer_model reuse.
    from open_optformer.optformer_searcher import load_optformer_model
    model, tokenizer, _ = load_optformer_model(
        args.checkpoint_dir, use_vllm=args.use_vllm,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    n_done, n_failed = 0, 0
    for benchmark_name, seed in itertools.product(benchmarks, seeds):
        out_path = args.out_dir / f"{benchmark_name}__seed{seed}.json"
        if out_path.exists():
            logger.info(f"Skipping {out_path.name} (already exists)")
            n_done += 1
            continue
        try:
            result = _run_one_rollout(
                benchmark_name, seed, args.checkpoint_dir, args.n_trials,
                args.use_vllm, args.gpu_memory_utilization, model=model, tokenizer=tokenizer,
            )
        except Exception:
            logger.exception(f"Rollout failed for benchmark={benchmark_name} seed={seed}")
            n_failed += 1
            continue
        with open(out_path, "w") as f:
            json.dump(result, f)
        n_done += 1
        logger.info(f"[{n_done + n_failed}/{len(benchmarks) * len(seeds)}] wrote {out_path.name}")

    logger.info(f"Done: {n_done} rollouts written, {n_failed} failed.")


if __name__ == "__main__":
    main()
