"""Precomputes each validation benchmark's TRUE (y_min, y_max) metric range
from the raw recorded experiment data under RESULTS_PATH, for the live
"debug quantization" eval ablation (see RUNNING.md / RUN_ORDER.md
"Debug quantization" sections, `experiments/quantization_debug/`).

Reuses `compute_true_metric_range_per_benchmark` -- the exact function the
training-time true-range ablation (`--metric_range_mode true` on
compile_data.py, see generate_dataset_all_true_range.sh) already uses to
compute this from the same `RESULTS_PATH` data -- rather than a second,
bespoke way of estimating a benchmark's true range live from its
blackbox/simulator. That also matters for correctness, not just reuse: the
live syne_tune scheduler (`SingleFidelityScheduler.on_trial_complete`)
already sign-flips max-mode metrics before `History` sees them, exactly
matching `compute_trial_metric_values`'s `if metric_mode == 'max': val =
-val` -- reading raw, un-flipped values from a blackbox table would have
been wrong for every max-mode benchmark.

Benchmark set: every entry in `validation_tasks.json` (the same file
compile_data.py uses to mark held-out validation tasks during training data
compilation) except the "hpob" family, per the debug-quantization eval's
scope. Fixes one known typo in that file: the "global_optimization_problems"
family lists "global-optimization_hartmann6" (double n), but the actual key
in benchmarks_definitions.py / syne_tune's global_optimization_problem_collection
is "hartman6" (single n) -- using the typo'd name here would silently match
zero recorded experiments and produce no range for that benchmark.
"""

import argparse
import itertools
import json
from pathlib import Path

from compile_data import compute_true_metric_range_per_benchmark
from load_data import get_metadata

EXCLUDED_FAMILIES = {"hpob"}
BENCHMARK_NAME_FIXES = {
    "global-optimization_hartmann6": "global-optimization_hartman6",
}


def load_validation_benchmarks(validation_tasks_path: Path) -> set:
    validation_tasks = json.load(open(validation_tasks_path))
    names = itertools.chain.from_iterable(
        v for k, v in validation_tasks.items() if k not in EXCLUDED_FAMILIES
    )
    return {BENCHMARK_NAME_FIXES.get(name, name) for name in names}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument(
        "--validation_tasks_json",
        type=str,
        default=str(Path(__file__).parent / "validation_tasks.json"),
    )
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--num_cores", type=int, default=None)
    args = parser.parse_args()

    results_path = Path(args.results_path)
    validation_benchmarks = load_validation_benchmarks(Path(args.validation_tasks_json))
    print(f"Computing true metric ranges for {len(validation_benchmarks)} validation benchmarks...")

    metadatas = get_metadata(root=results_path)
    metadatas = {
        name: metadata
        for name, metadata in metadatas.items()
        if metadata.get("benchmark") in validation_benchmarks
    }
    print(f"Found {len(metadatas)} recorded experiments across those benchmarks.")

    true_ranges = compute_true_metric_range_per_benchmark(
        metadatas, results_path, num_cores=args.num_cores
    )

    missing = validation_benchmarks - true_ranges.keys()
    if missing:
        print(f"[WARN] no recorded experiments found for {len(missing)} benchmarks: {sorted(missing)}")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({name: list(rng) for name, rng in true_ranges.items()}, f, indent=2)

    print(f"Wrote {len(true_ranges)} benchmark ranges to {out_path}")
