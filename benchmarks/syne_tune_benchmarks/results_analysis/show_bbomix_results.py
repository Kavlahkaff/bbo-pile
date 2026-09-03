"""
Visualize bbomix baselines / pretrained-OPT-CQR / bbomix finetuned-checkpoint results
together.

Uses OPT-CQR results run against the *pretrained* (non-finetuned) checkpoint
(bbomix_pretrained), not the earlier bbomix-opt directory -- that one's OPT-CQR runs
optimized a different objective (metric_valid_recon_loss, do_minimize=True) than
baselines/finetuned (metric_avg_ml_task_performance, do_minimize=False), making it an
unfair comparison. bbomix_pretrained's OPT-CQR runs target metric_avg_ml_task_performance
like everything else.

Restricts everything to the bbomix tasks present in the finetuned directory, forces the
common objective metric (metric_avg_ml_task_performance) across all three result sources,
drops any benchmark where a method has an implausibly low seed count (a sign that sweep
is still running on the cluster), and keeps the 3 finetuned checkpoints as distinct series
instead of merging them under one "OPT-best" label.

Produces two groups of figures under figures/bbomix/:
  - "all": RS, REA, BORE, TPE, CQR, OPT-CQR, and the 3 finetuned checkpoints together.
  - "checkpoints_only": just the 3 finetuned checkpoints, compared against each other.
"""
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dill
import numpy as np

from benchmarks.syne_tune_benchmarks.results_analysis.load_experiments_parallel import (
    load_benchmark_results,
)
from benchmarks.syne_tune_benchmarks.results_analysis.show_results import (
    figure_folder,
    stack_benchmark_results,
    generate_rank_results,
    plot_task_performance_over_time,
    plot_average_normalized_regret,
)
from syne_tune.util import catchtime

METRIC = "metric_avg_ml_task_performance"
NUM_TIME_STEPS = 100  # only visualize the first 100 trials
MAX_SEED = 30

BASELINE_METHODS = ["RS", "REA", "BORE", "TPE", "CQR"]
OPT_METHOD = "OPT-CQR"

BenchmarkResults = Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]]


def normalize_benchmark(name: str) -> str:
    """
    Maps benchmark names from the different bbomix result sources onto a common key.
    Baselines look like "bbomix-ontix-tcga-rna-clin-reactome-downstream" while
    finetuned/opt look like "bbomix-ontix-tcga-tcga-RNA-DNA-METH-CLIN-reactome"
    (duplicated cohort token, uppercase modalities, no "-downstream" suffix).
    """
    name = name.lower().replace("_", "-")
    if name.endswith("-downstream"):
        name = name[: -len("-downstream")]
    tokens = name.split("-")
    deduped = []
    for token in tokens:
        if not deduped or deduped[-1] != token:
            deduped.append(token)
    return "-".join(deduped)


def merge_into(combined: BenchmarkResults, source: BenchmarkResults, rename: Optional[Dict[str, str]] = None):
    for raw_benchmark, (t_range, method_dict) in source.items():
        key = normalize_benchmark(raw_benchmark)
        if rename:
            method_dict = {rename.get(m, m): v for m, v in method_dict.items()}
        if key not in combined:
            combined[key] = (t_range, dict(method_dict))
        else:
            existing_t_range, existing_methods = combined[key]
            assert len(existing_t_range) == len(t_range), (
                f"mismatched num_time_steps for {key}: {len(existing_t_range)} vs {len(t_range)}"
            )
            existing_methods.update(method_dict)


def model_size_label(checkpoint_dir_name: str) -> str:
    """e.g. "qwen3_80M_token_2B_lr_5e-3_..." -> "80M"."""
    match = re.search(r"qwen3_(\d+M)_", checkpoint_dir_name)
    assert match, f"could not extract model size from {checkpoint_dir_name!r}"
    return match.group(1)


def load_finetuned(finetuned_root: Path) -> Tuple[BenchmarkResults, List[str]]:
    checkpoint_dirs = sorted(finetuned_root.glob("step-*/*"))
    assert checkpoint_dirs, f"no checkpoint directories found under {finetuned_root}"

    combined: BenchmarkResults = {}
    task_sets = []
    checkpoint_labels = []
    for checkpoint_dir in checkpoint_dirs:
        step_name = checkpoint_dir.parent.name  # e.g. "step-00000731"
        label = f"Finetuned-{model_size_label(checkpoint_dir.name)}"
        checkpoint_labels.append(label)
        with catchtime(f"load finetuned checkpoint {step_name} ({label})"):
            results = load_benchmark_results(
                path=checkpoint_dir,
                methods=["OPT-best"],
                num_time_steps=NUM_TIME_STEPS,
                max_seed=MAX_SEED,
                metric_override=METRIC,
            )
        task_sets.append({normalize_benchmark(b) for b in results.keys()})
        merge_into(combined, results, rename={"OPT-best": label})

    common_tasks = set.union(*task_sets)
    for step_name, tasks in zip([d.parent.name for d in checkpoint_dirs], task_sets):
        if tasks != common_tasks:
            missing = common_tasks - tasks
            logging.warning(f"checkpoint {step_name} is missing tasks: {missing}")

    return combined, sorted(common_tasks)


def load_filtered(
    path: Path, methods: List[str], finetuned_tasks: List[str]
) -> BenchmarkResults:
    finetuned_tasks_set = set(finetuned_tasks)
    with catchtime(f"load {path}"):
        results = load_benchmark_results(
            path=path,
            methods=methods,
            num_time_steps=NUM_TIME_STEPS,
            max_seed=MAX_SEED,
            metric_override=METRIC,
            experiment_filter=lambda md: normalize_benchmark(md["benchmark"])
            in finetuned_tasks_set,
        )
    return results


def build_combined_results(
    baselines_path: Path, opt_path: Path, finetuned_path: Path
) -> Tuple[BenchmarkResults, List[str]]:
    finetuned_results, finetuned_tasks = load_finetuned(finetuned_path)
    print(f"finetuned dir covers {len(finetuned_tasks)} bbomix tasks: {finetuned_tasks}")

    opt_results = load_filtered(opt_path, [OPT_METHOD], finetuned_tasks)
    opt_matched = {normalize_benchmark(b) for b in opt_results.keys()}
    print(f"bbomix_pretrained matched {len(opt_matched)}/{len(finetuned_tasks)} finetuned tasks")
    missing_opt = set(finetuned_tasks) - opt_matched
    if missing_opt:
        logging.warning(f"bbomix_pretrained is missing tasks: {missing_opt}")

    baseline_results = load_filtered(baselines_path, BASELINE_METHODS, finetuned_tasks)
    baseline_matched = {normalize_benchmark(b) for b in baseline_results.keys()}
    print(
        f"bbomix_baselines matched {len(baseline_matched)}/{len(finetuned_tasks)} finetuned tasks"
    )
    missing_baseline = set(finetuned_tasks) - baseline_matched
    if missing_baseline:
        logging.warning(f"bbomix_baselines is missing tasks: {missing_baseline}")

    checkpoint_labels = sorted(
        {m for _, method_dict in finetuned_results.values() for m in method_dict.keys()}
    )

    combined: BenchmarkResults = {}
    merge_into(combined, finetuned_results)
    merge_into(combined, opt_results)
    merge_into(combined, baseline_results)

    # stack_benchmark_results purges a method from ALL benchmarks' plots if it's
    # simply missing from a single one (e.g. a run that's still in progress, or
    # whose results.csv.zip failed to write on the cluster) -- so a benchmark
    # missing any expected method must be dropped entirely rather than letting
    # that one gap silently erase the method everywhere else.
    expected_methods = set(BASELINE_METHODS) | {OPT_METHOD} | set(checkpoint_labels)
    incomplete_benchmarks = []
    for benchmark, (t_range, method_dict) in combined.items():
        missing = expected_methods - method_dict.keys()
        if missing:
            logging.warning(
                f"{benchmark}: missing methods {sorted(missing)} (likely still "
                f"running on the cluster) -- dropping this benchmark from the comparison"
            )
            incomplete_benchmarks.append(benchmark)
    for benchmark in incomplete_benchmarks:
        del combined[benchmark]

    # each source was independently sliced to its own min seed count by
    # convert_all_to_numpy, so after merging, methods for the same benchmark can
    # have differing seed counts (e.g. baselines' BORE with 29 vs opt/finetuned's
    # 30) -- reconcile to a common min per benchmark so np.stack works downstream.
    for benchmark, (t_range, method_dict) in combined.items():
        seed_counts = {m: v.shape[0] for m, v in method_dict.items()}
        min_seeds = min(seed_counts.values())
        if len(set(seed_counts.values())) > 1:
            logging.warning(
                f"{benchmark}: mismatched seed counts {seed_counts}, slicing all to {min_seeds}"
            )
        for method in method_dict:
            method_dict[method] = method_dict[method][:min_seeds]

    return combined, checkpoint_labels


def load_and_cache_combined(
    baselines_path: Path,
    opt_path: Path,
    finetuned_path: Path,
    cache_dir: Path,
    reuse_cache: bool,
) -> Tuple[BenchmarkResults, List[str]]:
    cache_file = cache_dir / "bbomix-combined-results-cache.dill"
    if reuse_cache and cache_file.exists():
        with catchtime(f"loading cached combined results from {cache_file}"):
            with open(cache_file, "rb") as f:
                return dill.load(f)

    combined, checkpoint_labels = build_combined_results(baselines_path, opt_path, finetuned_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        dill.dump((combined, checkpoint_labels), f)
    return combined, checkpoint_labels


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baselines_path", type=str, default="~/bbomix_baselines/results_downstream"
    )
    parser.add_argument(
        "--opt_path",
        type=str,
        default="~/bbomix_pretrained/final/qwen3_80M_token_2B_lr_5e-3_bsz_16_seed_0",
    )
    parser.add_argument(
        "--finetuned_path", type=str, default="~/bbomix_finetuned/results_bbomix"
    )
    parser.add_argument("--reuse_cache", action="store_true")
    args, _ = parser.parse_known_args()

    baselines_path = Path(args.baselines_path).expanduser()
    opt_path = Path(args.opt_path).expanduser()
    finetuned_path = Path(args.finetuned_path).expanduser()
    for p in [baselines_path, opt_path, finetuned_path]:
        assert p.exists(), f"path does not exist: {p}"

    cache_dir = Path("figures") / "bbomix"
    combined_results, checkpoint_labels = load_and_cache_combined(
        baselines_path, opt_path, finetuned_path, cache_dir, args.reuse_cache
    )
    assert len(combined_results) > 0, "no bbomix results loaded"

    groups = {
        "all": BASELINE_METHODS + [OPT_METHOD] + checkpoint_labels,
        "checkpoints_only": checkpoint_labels,
    }
    benchmark_families = ["bbomix"]

    for group_name, methods in groups.items():
        result_folder = figure_folder(Path("figures") / "bbomix" / group_name)
        # stack_benchmark_results mutates methods_to_show in place (removes methods
        # missing from some benchmark) -- pass the same list object through so later
        # calls below stay in sync with the reduced set, matching show_results.py's
        # own __main__ pattern.
        stacked_benchmark_results = stack_benchmark_results(
            benchmark_results_dict=combined_results,
            methods_to_show=methods,
            benchmark_families=benchmark_families,
        )
        if len(stacked_benchmark_results) == 0:
            print(f"skipping group {group_name}: no methods left after stacking filter")
            continue

        rename_dict = {}
        with catchtime(f"generating rank table ({group_name})"):
            generate_rank_results(
                stacked_benchmark_results=stacked_benchmark_results,
                benchmark_families=benchmark_families,
                methods_to_show=methods,
                rename_dict=rename_dict,
                result_folder=result_folder,
                legend_outside=True,
            )

        with catchtime(f"generating plots per task ({group_name})"):
            plot_task_performance_over_time(
                benchmark_results=combined_results,
                methods_to_show=methods,
                rename_dict=rename_dict,
                result_folder=result_folder,
                plot_regret=False,
                legend_outside=True,
            )

        plot_average_normalized_regret(
            stacked_benchmark_results=stacked_benchmark_results,
            methods_to_show=methods,
            rename_dict=rename_dict,
            result_folder=result_folder,
            title="Normalized-regret",
            legend_outside=True,
        )
