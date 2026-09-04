"""Approach C (expert iteration / STaR) -- step 2: score + select rollouts.

Consumes the per-rollout JSON files written by generate_self_play_rollouts.py
and train_best_baselines.json (compute_train_best_baselines.sh's output: the
"best of the six baseline optimizers" mean_val threshold per training task).
A rollout is kept iff its realized (final-value, min-convention) regret beats
that threshold -- i.e. the model's OWN self-generated trajectory, scored
against the real surrogate, outperformed the best of RS/REA/BORE/TPE/CQR/HEBO
on that same task. Kept rollouts are serialized into the same
{"segments": [[text, weight], ...]} JSONL format
open_optformer/training/advantage_sft_data.py already reads, with uniform
weight=1.0 per trial (plain SFT on kept trajectories -- see
History.get_prompt_with_weights and compile_data.py::process_metadata_with_advantage,
whose JSONL shape this mirrors).

Round 1 selection is whole-rollout only (one scalar regret per rollout, keep
or discard the entire trajectory) -- see the plan's "Selection granularity"
decision. Per-step sub-trajectory selection is a documented future extension,
not implemented here.

Usage:
    python select_self_play_rollouts.py \\
        --rollout_dir /path/to/self_play/round_1/qwen3_2M/rollouts \\
        --best_baselines_json train_best_baselines.json \\
        --round 1 \\
        --out_train_jsonl /path/to/self_play/round_1/qwen3_2M/self_play_train.jsonl \\
        --out_valid_jsonl /path/to/self_play/round_1/qwen3_2M/self_play_valid.jsonl \\
        [--prior_train_jsonl /path/to/self_play/round_0/.../self_play_train.jsonl ...]
"""
import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def _realized_regret(trials: list[dict]) -> float:
    """Final value of the rollout, min-convention (mode_sign already applied
    at generation time -- see generate_self_play_rollouts.py). Matches
    find_best_baselines.py's scoring so the comparison against mean_val is
    apples-to-apples."""
    return trials[-1]["y"]


def _rollout_to_segments(rollout: dict, config_space: dict, remove_names: bool = True):
    """Rebuild a History from a serialized rollout and return its
    get_prompt_with_weights segments, with uniform weight=1.0 (plain SFT on
    every kept trial -- no per-trial reweighting, unlike Approach B)."""
    from open_optformer.history import History

    hist = History(
        config_space=config_space,
        name=rollout["benchmark"],
        algorithm="self-play",
        metric_names=[rollout["metric"]],
        remove_names=remove_names,
    )
    for trial in rollout["trials"]:
        hist.add_trial(trial["config"], trial["y"])
    _, segments = hist.get_prompt_with_weights(base_weight=1.0)
    return segments


def select_and_serialize(rollout_dir: Path, best_baselines_json: Path,
                          round_num: int, out_train_jsonl: Path, out_valid_jsonl: Path,
                          valid_frac: float = 0.1, prior_train_jsonls=(), seed: int = 0):
    with open(best_baselines_json) as f:
        best_baselines = json.load(f)

    kept, n_total = [], 0
    n_kept_by_benchmark = defaultdict(int)
    n_total_by_benchmark = defaultdict(int)

    from open_optformer.sample_distribution import _load_blackbox
    # Rollout files are named "{benchmark}__seed{seed}.json" and iterated in
    # sorted order, so every seed of a given benchmark is contiguous -- cache
    # the loaded blackbox by benchmark name instead of reloading it (an
    # expensive surrogate load) for every one of the n_seeds rollout files.
    _blackbox_cache: dict = {}

    def _get_blackbox(benchmark_name):
        if benchmark_name not in _blackbox_cache:
            bb, _ = _load_blackbox(benchmark_name)
            _blackbox_cache[benchmark_name] = bb
        return _blackbox_cache[benchmark_name]

    for rollout_path in sorted(rollout_dir.glob("*.json")):
        with open(rollout_path) as f:
            rollout = json.load(f)
        benchmark = rollout["benchmark"]
        n_total += 1
        n_total_by_benchmark[benchmark] += 1

        if benchmark not in best_baselines:
            logger.warning(f"No best-baseline threshold for {benchmark!r}, skipping "
                            f"{rollout_path.name} (did you run compute_train_best_baselines.sh "
                            f"with the same --all_training_tasks_json?)")
            continue

        threshold = best_baselines[benchmark]["mean_val"]
        regret = _realized_regret(rollout["trials"])
        if regret >= threshold:
            continue  # did not beat the best of the six baselines -- discard

        bb = _get_blackbox(benchmark)
        segments = _rollout_to_segments(rollout, bb.configuration_space)

        kept.append({
            "segments": segments,
            "experiment_name": f"{benchmark}__seed{rollout['seed']}__round{round_num}",
            "benchmark": benchmark,
            "algorithm": "self-play",
        })
        n_kept_by_benchmark[benchmark] += 1

    n_kept = len(kept)
    logger.info(f"Round {round_num}: kept {n_kept}/{n_total} rollouts "
                f"({100 * n_kept / max(n_total, 1):.1f}%)")
    for benchmark in sorted(n_total_by_benchmark):
        k, t = n_kept_by_benchmark[benchmark], n_total_by_benchmark[benchmark]
        logger.info(f"  {benchmark}: {k}/{t} ({100 * k / t:.1f}%)")

    rng = random.Random(seed)
    rng.shuffle(kept)
    n_valid = max(1, int(len(kept) * valid_frac)) if kept else 0
    new_valid, new_train = kept[:n_valid], kept[n_valid:]

    prior_train = []
    for prior_path in prior_train_jsonls:
        with open(prior_path) as f:
            prior_train.extend(json.loads(line) for line in f if line.strip())
    logger.info(f"Cumulative train pool: {len(prior_train)} prior + {len(new_train)} "
                f"new = {len(prior_train) + len(new_train)} examples")

    out_train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_train_jsonl, "w") as f:
        for example in prior_train + new_train:
            f.write(json.dumps(example) + "\n")
    with open(out_valid_jsonl, "w") as f:
        for example in new_valid:
            f.write(json.dumps(example) + "\n")

    return n_kept, n_total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout_dir", type=Path, required=True)
    p.add_argument("--best_baselines_json", type=Path, required=True)
    p.add_argument("--round", type=int, required=True, dest="round_num")
    p.add_argument("--out_train_jsonl", type=Path, required=True)
    p.add_argument("--out_valid_jsonl", type=Path, required=True)
    p.add_argument("--valid_frac", type=float, default=0.1)
    p.add_argument("--prior_train_jsonl", type=Path, nargs="*", default=(),
                    help="Prior rounds' train JSONL(s), unioned in for the "
                         "cumulative-pool checkpoint-lineage strategy (see plan).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s")

    select_and_serialize(
        rollout_dir=args.rollout_dir,
        best_baselines_json=args.best_baselines_json,
        round_num=args.round_num,
        out_train_jsonl=args.out_train_jsonl,
        out_valid_jsonl=args.out_valid_jsonl,
        valid_frac=args.valid_frac,
        prior_train_jsonls=args.prior_train_jsonl,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
