"""Build `all_training_tasks.json`: the actual training-task universe.

Neither `hpo_tasks.json` nor `validation_tasks.json` is a train-task list --
`hpo_tasks.json` is not read by any code in this repo (verified by grep), and
`validation_tasks.json` is only ever used by `compile_data.py` to mark
benchmarks as held-out (`is_valid`). The real training-task universe is every
key in each family's `*_benchmark_definitions` dict (the full set of tasks
`benchmarks/syne_tune_benchmarks/*_benchmarks.py` knows how to build a
blackbox+surrogate for), minus whatever `validation_tasks.json` actually
excludes for that family.

Verified counts (see plan): fcnet=3, nas201=2, lcbench=30, pd1=20,
tabrepo=1659, hpob=917 -- 2631 total.

`hpob` caveat: 44 of the 57 `hpob` entries in `validation_tasks.json` use the
typo'd prefix `hpob_4976_...` instead of the real `hpob_4796_...`, so
`compile_data.py`'s exact-string `is_valid` check never actually excludes
those 44 tasks -- only 13 of the intended 57 hpob held-out tasks are
genuinely held out today. This is a pre-existing bug in the data already used
to pretrain the checkpoints. This script deliberately replicates that
behavior (exact-string matching, typo included) rather than fixing it, so
Approach C's self-play rollout scope stays consistent with what the existing
checkpoints were actually trained/held-out on. Do not "fix" the typo here --
that is a separate, out-of-scope cleanup.

`global_optimization_problems` is excluded entirely: it uses a live-function
AskTellScheduler interface, not the `load_blackbox`/`add_surrogate` interface
the other families (and Approach C's rollout generation) rely on.

Usage:
    python build_all_training_tasks.py \\
        --validation_tasks_json validation_tasks.json \\
        --out_json all_training_tasks.json
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

FAMILY_MODULES = {
    "fcnet": ("fcnet_benchmarks", "fcnet_benchmark_definitions"),
    "nas201": ("nas201_benchmarks", "nas201_benchmark_definitions"),
    "lcbench": ("lcbench_benchmarks", "lcbench_benchmark_definitions"),
    "pd1": ("pd1_benchmarks", "pd1_benchmark_definitions"),
    "tabrepo": ("tabrepo_benchmarks", "tabrepo_benchmark_definitions"),
    "hpob": ("hpob_benchmarks", "hpob_benchmark_definitions"),
}


def build_all_training_tasks(validation_tasks_json: Path) -> dict:
    syne_tune_benchmarks_dir = str(
        Path(__file__).resolve().parent.parent / "benchmarks" / "syne_tune_benchmarks"
    )
    if syne_tune_benchmarks_dir not in sys.path:
        sys.path.insert(0, syne_tune_benchmarks_dir)

    with open(validation_tasks_json) as f:
        validation_tasks = json.load(f)

    all_training_tasks = {}
    for family, (module_name, dict_name) in FAMILY_MODULES.items():
        module = importlib.import_module(module_name)
        full_universe = list(getattr(module, dict_name).keys())
        held_out = set(validation_tasks.get(family, []))
        all_training_tasks[family] = [t for t in full_universe if t not in held_out]

    return all_training_tasks


def main() -> None:
    default_validation_json = Path(__file__).resolve().parent / "validation_tasks.json"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation_tasks_json", type=Path, default=default_validation_json)
    p.add_argument(
        "--out_json",
        type=Path,
        default=Path(__file__).resolve().parent / "all_training_tasks.json",
    )
    args = p.parse_args()

    all_training_tasks = build_all_training_tasks(args.validation_tasks_json)

    for family, tasks in all_training_tasks.items():
        print(f"{family}: {len(tasks)} training tasks")
    print(f"total: {sum(len(v) for v in all_training_tasks.values())} training tasks")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(all_training_tasks, f, indent=1)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
