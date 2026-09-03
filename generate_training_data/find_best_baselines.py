"""Find, for each benchmark, the reference algorithm with the best AVERAGE
final value across all its recorded seeds, and index every one of that
algorithm's recorded seed-runs by seed.

This is a thin, run-once CLI around `compile_data.py`'s existing
`find_best_algorithms` (the same seed-averaging selection logic used by
`--only_best --keep_all_seeds_of_best`), so "best baseline" is computed
identically here and when building training data. The output JSON is a
reusable, on-disk cache: `open_optformer/sample_best_baseline_comparison.py`
reads it instead of re-scanning the (large) raw results directory on every
run.

  python find_best_baselines.py \\
      --results_path /data/horse/ws/luth474h-master_thesis/raw_data_bbo_pile \\
      --out_json best_baselines.json
"""
import argparse
import itertools
import json
from pathlib import Path

from compile_data import find_best_algorithms
from load_data import get_metadata

DEFAULT_METHODS = ["REA", "TPE", "BORE", "CQR", "RS", "HEBO"]


def _flatten_tasks_json(tasks_json: Path) -> list[str]:
    with open(tasks_json) as f:
        families = json.load(f)
    return list(itertools.chain.from_iterable(families.values()))


if __name__ == "__main__":
    default_validation_json = Path(__file__).resolve().parent / "validation_tasks.json"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_path", type=Path, required=True,
                    help="Root of the raw syne-tune results (raw_data_bbo_pile).")
    p.add_argument("--benchmark", type=str, nargs="+", default=None,
                    dest="benchmarks",
                    help="Benchmarks to compute a best-baseline entry for. "
                         "Defaults to every benchmark in --validation_tasks_json "
                         "(or --all_training_tasks_json, if given).")
    p.add_argument("--validation_tasks_json", type=Path, default=default_validation_json,
                    help="Used only when --benchmark and --all_training_tasks_json "
                         "are both omitted.")
    p.add_argument("--all_training_tasks_json", type=Path, default=None,
                    help="Use generate_training_data/all_training_tasks.json (or an "
                         "equivalent file, same {family: [task, ...]} shape) as the "
                         "benchmark source instead of --validation_tasks_json -- "
                         "needed for Approach C (self-play), which scores rollouts "
                         "generated on TRAINING tasks, not the held-out set this "
                         "script defaults to. Takes precedence over "
                         "--validation_tasks_json when --benchmark is omitted.")
    p.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS,
                    help="Reference algorithms eligible to be 'best' "
                         "(matches compile_data.py's whitelist).")
    p.add_argument("--out_json", type=Path, required=True)
    args = p.parse_args()

    tasks_json = args.all_training_tasks_json or args.validation_tasks_json
    benchmarks = args.benchmarks or _flatten_tasks_json(tasks_json)
    print(f"Benchmarks ({len(benchmarks)}): {benchmarks}")

    print(f"Scanning metadata under {args.results_path} ...")
    metadatas = get_metadata(root=args.results_path)
    methods = set(args.methods)
    benchmarks_set = set(benchmarks)
    metadatas = {
        k: v for k, v in metadatas.items()
        if v.get("algorithm") in methods
        and v.get("benchmark", v.get("entrypoint", "unknown")) in benchmarks_set
    }
    print(f"{len(metadatas)} experiment metadata entries match the requested "
          f"benchmarks/methods.")

    best_by_benchmark = find_best_algorithms(metadatas, args.results_path)

    missing = benchmarks_set - best_by_benchmark.keys()
    if missing:
        print(f"WARNING: no matching experiments found for: {sorted(missing)}")

    out = {
        bench: {
            "algorithm": r["algorithm"],
            "mean_val": r["mean_val"],
            "seed_experiments": {
                str(seed): name for seed, name in r["seed_experiments"].items()
            },
        }
        for bench, r in best_by_benchmark.items()
        if bench in benchmarks_set
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out_json} ({len(out)} benchmarks)")
