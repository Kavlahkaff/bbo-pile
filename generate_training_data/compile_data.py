import logging
import json
import os
import tqdm
import random
import itertools
import numpy as np
import multiprocessing
import sys

from pathlib import Path
from argparse import ArgumentParser
from syne_tune.util import catchtime

from load_data import get_metadata, create_history_from_results

logger = logging.getLogger(__name__)

def process_best_trajectory(args_tuple):
    name, metadata, path = args_tuple
    from load_data import load_result, get_config_space_from_metadata
    benchmark_name = metadata.get('benchmark', metadata.get('entrypoint', 'unknown'))
    metric_name = metadata["metric_names"][0]
    metric_mode = metadata.get('metric_mode', 'min')

    try:
        config_space = get_config_space_from_metadata(metadata)
        res = load_result(name, metric_name, config_space, path)
    except Exception:
        res = None

    val = None
    if res is not None and metric_name in res.columns:
        if metric_mode == 'max':
            val = -res[metric_name].max()
        else:
            val = res[metric_name].min()
    return benchmark_name, name, val

def find_best_algorithms(metadatas: dict, path: Path,
                          num_cores: int = None) -> dict:
    """For each benchmark present in `metadatas`, determine:
      - "best_experiment": (experiment_name, val), the single experiment
        with the lowest `val` (see `process_best_trajectory`) across every
        algorithm/seed for that benchmark.
      - "algorithm"/"mean_val": the algorithm with the lowest MEAN `val`
        across all of its recorded seeds for that benchmark (i.e. best
        average final value over all seeds).
      - "seed_experiments": {seed: experiment_name}, every recorded seed of
        that winning algorithm for that benchmark.

    Reused by both `--keep_all_seeds_of_best` (via "algorithm") and the
    single-best-experiment path (via "best_experiment") below, and by
    `find_best_baselines.py`.
    """
    from collections import defaultdict

    tasks = [(name, metadata, path) for name, metadata in metadatas.items()]

    if num_cores is None:
        try:
            num_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cores = multiprocessing.cpu_count()

    logger.info(f"Detecting best trajectories using {num_cores} cores for {len(tasks)} tasks...")

    best_experiments = {}
    benchmark_algo_vals = defaultdict(lambda: defaultdict(list))
    benchmark_algo_seed_experiments = defaultdict(lambda: defaultdict(dict))

    with multiprocessing.Pool(processes=num_cores) as pool:
        for benchmark_name, name, val in tqdm.tqdm(
                pool.imap_unordered(process_best_trajectory, tasks),
                total=len(tasks),
                desc="Finding best trajectories",
                mininterval=5.0):
            if val is None:
                continue

            if benchmark_name not in best_experiments or val < best_experiments[benchmark_name][1]:
                best_experiments[benchmark_name] = (name, val)

            algo = metadatas[name]["algorithm"]
            seed = metadatas[name].get("seed")
            benchmark_algo_vals[benchmark_name][algo].append(val)
            benchmark_algo_seed_experiments[benchmark_name][algo][seed] = name

    results = {}
    for bench, algos in benchmark_algo_vals.items():
        best_algo = None
        best_avg = float('inf')
        for algo, vals in algos.items():
            avg_val = np.mean(vals)
            if avg_val < best_avg:
                best_avg = avg_val
                best_algo = algo
        results[bench] = {
            "algorithm": best_algo,
            "mean_val": best_avg,
            "seed_experiments": benchmark_algo_seed_experiments[bench][best_algo],
            "best_experiment": best_experiments.get(bench),
        }
    return results


def compute_trial_running_best(args_tuple):
    """Per-experiment running-best-so-far metric value at each trial index,
    in the same min-convention and the same per-trial ordering (groupby
    'trial_id', last row per trial, ascending trial_id) as
    `History.from_syne_tune_experiment` uses when building `hist.trials` --
    so the returned list stays index-aligned with a `History`'s trials for
    the same experiment.

    This is additive, advantage-reweighted-SFT-only machinery: it does not
    modify or get called by `process_best_trajectory`/`find_best_algorithms`
    or the `--only_best`/`--rename_best` selection pipeline above, even
    though it shares the same "how good is this trajectory" spirit.
    """
    name, metadata, path = args_tuple
    from load_data import load_result, get_config_space_from_metadata
    benchmark_name = metadata.get('benchmark', metadata.get('entrypoint', 'unknown'))
    metric_name = metadata["metric_names"][0]
    metric_mode = metadata.get('metric_mode', 'min')

    try:
        config_space = get_config_space_from_metadata(metadata)
        res = load_result(name, metric_name, config_space, path)
    except Exception:
        res = None

    if res is None or metric_name not in res.columns:
        return benchmark_name, name, None

    per_trial_vals = []
    for _, trial in res.groupby('trial_id'):
        row = trial.iloc[-1]
        val = row[metric_name]
        if metric_mode == 'max':
            val = -val
        per_trial_vals.append(val)

    if len(per_trial_vals) == 0:
        return benchmark_name, name, None

    running_best = np.minimum.accumulate(per_trial_vals).tolist()
    return benchmark_name, name, running_best


def compute_peer_baseline_per_benchmark(metadatas: dict, path: Path, num_cores: int = None) -> dict:
    """For each benchmark, the mean running-best-so-far curve across ALL
    optimizers/seeds/experiments recorded for that benchmark in `metadatas`
    (short trajectories are forward-filled to the longest trajectory's
    length before averaging). This is the peer baseline that per-trial
    advantages are computed against.
    """
    tasks = [(name, metadata, path) for name, metadata in metadatas.items()]

    if num_cores is None:
        try:
            num_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cores = multiprocessing.cpu_count()

    logger.info(f"Computing peer baselines using {num_cores} cores for {len(tasks)} tasks...")

    per_benchmark_curves = {}
    with multiprocessing.Pool(processes=num_cores) as pool:
        for benchmark_name, _, running_best in tqdm.tqdm(
                pool.imap_unordered(compute_trial_running_best, tasks),
                total=len(tasks),
                desc="Computing per-trial running-best curves",
                mininterval=5.0):
            if not running_best:
                continue
            per_benchmark_curves.setdefault(benchmark_name, []).append(running_best)

    peer_baseline = {}
    for benchmark_name, curves in per_benchmark_curves.items():
        max_len = max(len(c) for c in curves)
        padded = np.empty((len(curves), max_len))
        for i, c in enumerate(curves):
            arr = np.array(c, dtype=float)
            if len(arr) < max_len:
                arr = np.concatenate([arr, np.full(max_len - len(arr), arr[-1])])
            padded[i] = arr
        peer_baseline[benchmark_name] = padded.mean(axis=0)
    return peer_baseline


def compute_trial_metric_values(args_tuple):
    """Per-experiment list of RAW per-trial metric values (not the
    running-best curve -- that's monotone and understates the bad/worse
    tail's true max). Same min-convention/per-trial ordering as
    `compute_trial_running_best`. Debug/ablation-only: feeds
    `compute_true_metric_range_per_benchmark`, used by `--metric_range_mode
    true` to quantize training data against each benchmark's real observed
    range instead of the causal default (see History.metric_range_override).
    """
    name, metadata, path = args_tuple
    from load_data import load_result, get_config_space_from_metadata
    benchmark_name = metadata.get('benchmark', metadata.get('entrypoint', 'unknown'))
    metric_name = metadata["metric_names"][0]
    metric_mode = metadata.get('metric_mode', 'min')

    try:
        config_space = get_config_space_from_metadata(metadata)
        res = load_result(name, metric_name, config_space, path)
    except Exception:
        res = None

    if res is None or metric_name not in res.columns:
        return benchmark_name, None

    per_trial_vals = []
    for _, trial in res.groupby('trial_id'):
        row = trial.iloc[-1]
        val = row[metric_name]
        if metric_mode == 'max':
            val = -val
        per_trial_vals.append(float(val))

    if len(per_trial_vals) == 0:
        return benchmark_name, None

    return benchmark_name, per_trial_vals


def compute_true_metric_range_per_benchmark(metadatas: dict, path: Path, num_cores: int = None,
                                             percentile_lo: float = 0.0, percentile_hi: float = 95.0) -> dict:
    """For each benchmark, the TRUE (oracle) (min, max) metric quantization
    range, percentile-clipped (same (0, 95) defaults as
    `History.metric_percentile_lo/_hi`'s causal range) over the RAW per-trial
    metric values pooled across ALL optimizers/seeds/experiments recorded for
    that benchmark in `metadatas` -- as opposed to the causal per-trial range
    `History` uses by default. Debug/ablation only (`--metric_range_mode
    true`): lets you test whether a gap to baselines is a quantization
    artifact rather than a model-capability issue.

    Percentile-clipped rather than raw min/max: a single diverged trial
    anywhere in the pool would otherwise consume most of the oracle range's
    quantization resolution, which would make an "oracle vs. causal"
    comparison confound range *source* with range *robustness*. Pass
    percentile_lo=0, percentile_hi=100 for the raw extent instead.
    """
    tasks = [(name, metadata, path) for name, metadata in metadatas.items()]

    if num_cores is None:
        try:
            num_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cores = multiprocessing.cpu_count()

    logger.info(f"Computing true per-benchmark metric ranges using {num_cores} cores for {len(tasks)} tasks...")

    from collections import defaultdict
    pooled_vals = defaultdict(list)
    with multiprocessing.Pool(processes=num_cores) as pool:
        for benchmark_name, vals in tqdm.tqdm(
                pool.imap_unordered(compute_trial_metric_values, tasks),
                total=len(tasks),
                desc="Computing true per-benchmark metric ranges",
                mininterval=5.0):
            if vals is None:
                continue
            pooled_vals[benchmark_name].extend(vals)

    true_range = {}
    for benchmark_name, vals in pooled_vals.items():
        y_min = float(np.percentile(vals, percentile_lo))
        y_max = float(np.percentile(vals, percentile_hi))
        if y_min == y_max:
            y_max += 1  # avoid division by zero in quantization, matches History
        true_range[benchmark_name] = (y_min, y_max)
    return true_range


def compute_advantages_for_experiment(name, metadata, path, peer_baseline_by_benchmark, temperature=None):
    """Per-trial advantage = peer_baseline[benchmark][t] - running_best_this_experiment[t]
    (min-convention, so a trajectory doing better than its peers at trial t
    gets a positive advantage). If `temperature` is given, returns
    `exp(advantage / temperature)` instead of the raw difference.
    """
    benchmark_name, _, running_best = compute_trial_running_best((name, metadata, path))
    if running_best is None or benchmark_name not in peer_baseline_by_benchmark:
        return None

    baseline = peer_baseline_by_benchmark[benchmark_name]
    n = len(running_best)
    if len(baseline) < n:
        baseline = np.concatenate([baseline, np.full(n - len(baseline), baseline[-1])])
    advantages = baseline[:n] - np.array(running_best)

    if temperature is not None:
        advantages = np.exp(advantages / temperature)

    return advantages.tolist()


def process_metadata_with_advantage(args_tuple):
    """Builds one advantage-weighted training example for a single experiment:
    a `History.get_prompt_with_weights` segments list, where each trial's
    span is weighted by that trial's advantage relative to its benchmark's
    peer baseline. Additive/parallel to `process_metadata` above -- does not
    call or modify it.
    """
    name, metadata, path, peer_baseline_by_benchmark, temperature, is_valid, metric_range_by_benchmark = args_tuple
    from load_data import load_result, get_config_space_from_metadata
    from open_optformer.history import History
    from syne_tune.experiments import ExperimentResult

    try:
        advantages = compute_advantages_for_experiment(
            name, metadata, path, peer_baseline_by_benchmark, temperature=temperature
        )
        if advantages is None:
            return is_valid, None

        config_space = get_config_space_from_metadata(metadata)
        metric_name = metadata["metric_names"][0]
        res = load_result(name, metric_name, config_space, path)
        benchmark_name_lookup = metadata.get('benchmark', metadata.get('entrypoint', 'unknown'))
        metric_range_override = (
            metric_range_by_benchmark.get(benchmark_name_lookup) if metric_range_by_benchmark else None
        )
        hist = History.from_syne_tune_experiment(
            ExperimentResult(name=name, metadata=metadata, results=res, path=path, tuner=None),
            metric_range_override=metric_range_override,
        )
        advantages = advantages[:len(hist.trials)]
        _, segments = hist.get_prompt_with_weights(advantages=advantages)

        benchmark_name = metadata.get('benchmark', metadata.get('entrypoint', 'unknown'))
        example = {
            "segments": segments,
            "experiment_name": name,
            "benchmark": benchmark_name,
            "algorithm": metadata.get("algorithm"),
        }
        return is_valid, example
    except Exception as e:
        logger.error(f"Error computing advantage-weighted example for {name}: {e}")
        return is_valid, None


def process_metadata(args_tuple):
    (name, metadata, path, max_num_trials, remove_names, num_permutation,
     sample_shorter_trajectories, is_valid, metric_range_by_benchmark) = args_tuple
    from load_data import create_history_from_results
    histories = []

    benchmark_name_lookup = metadata.get('benchmark', metadata.get('entrypoint', 'unknown'))
    metric_range_override = (
        metric_range_by_benchmark.get(benchmark_name_lookup) if metric_range_by_benchmark else None
    )

    try:
        histories.extend(create_history_from_results(name, metadata, path, max_num_trials,
                                                     remove_names=remove_names,
                                                     n_permutation=num_permutation,
                                                     metric_range_override=metric_range_override))
        if not is_valid and sample_shorter_trajectories:
            for mt in [1, 5, 10, 20]:
                histories.extend(create_history_from_results(name, metadata, path,
                                                             mt,
                                                             remove_names=remove_names,
                                                             metric_range_override=metric_range_override,
                                                             n_permutation=0))
    except Exception as e:
        logger.error(f"Error processing {name}: {e}")
        
    return is_valid, histories

if __name__ == "__main__":
    class FlushingStreamHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(levelname)s] %(message)s', 
        force=True, 
        handlers=[FlushingStreamHandler(sys.stdout)]
    )

    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="path where to find the results",
    )
    parser.add_argument(
        "--max_seed",
        type=int,
        required=False,
        default=30,
    )
    parser.add_argument(
        "--sample_shorter_trajectories",
        action='store_true',
        help="additionally add just the first [1, 5, 10, 20] trials of the trajectory",
    )
    parser.add_argument(
        "--num_permutation",
        type=int,
        required=False,
        default=5,
    )
    parser.add_argument(
        "--permutation_config",
        type=str,
        required=False,
        help="Path to JSON file mapping benchmark categories to permutation counts",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="path to store the results",
    )
    parser.add_argument(
        "--remove_names",
        action='store_true',
        help="remove names of benchmark and hypers",
    )
    parser.add_argument(
        "--only_best",
        action='store_true',
        help="only keep the best performing algorithm trajectory for each blackbox task",
    )
    parser.add_argument(
        "--rename_best",
        action='store_true',
        help="replace the algorithm name with 'best' when using --only_best",
    )
    parser.add_argument(
        "--keep_all_seeds_of_best",
        action='store_true',
        help="when using --only_best, keep all seeds of the best performing algorithm for each benchmark",
    )
    parser.add_argument(
        "--emit_advantage_weighted",
        action='store_true',
        help="additionally write advantage_train.jsonl/advantage_valid.jsonl for advantage-reweighted SFT. "
             "Runs over the full metadata pool BEFORE --only_best/--rename_best filtering (if those are also "
             "passed), independent of that selection pipeline: this is a separate selection philosophy that "
             "reweights every trajectory's trials by advantage relative to its benchmark's peers, rather than "
             "keeping/renaming a single best trajectory.",
    )
    parser.add_argument(
        "--advantage_temperature",
        type=float,
        required=False,
        default=None,
        help="if set, per-trial advantage weights are exp(advantage / temperature) instead of the raw "
             "difference against the peer baseline",
    )
    parser.add_argument(
        "--metric_range_mode",
        type=str,
        choices=["causal", "true"],
        default="causal",
        help="Debug/ablation knob for History's metric quantization range. 'causal' (default) computes "
             "each trial's range only from trials observed up to that point (matches inference-time "
             "availability). 'true' instead quantizes every trial in a benchmark against that "
             "benchmark's TRUE (min, max) metric value observed across ALL recorded experiments -- an "
             "oracle range unavailable at real inference time, useful for testing whether a gap to "
             "baselines is a quantization artifact rather than a model-capability issue.",
    )

    methods = [
        "REA",
        "TPE",
        "BORE",
        "CQR",
        "RS",
        "HEBO",
    ]

    args, _ = parser.parse_known_args()
    logger.info(f"Starting data compilation. Reading from '{args.path}', Output to '{args.output_path}'.")

    assert Path(args.path).exists()
    max_seed = args.max_seed
    max_num_trials = 100

    path = Path(args.path)
    output_path = Path(args.output_path)
    os.makedirs(output_path, exist_ok=True)
    experiment_filter = None

    validation_tasks = json.load(open('validation_tasks.json'))
    validation_tasks = list(itertools.chain.from_iterable(validation_tasks.values()))

    with catchtime("load benchmark results"):

        with catchtime("Load metadata"):
            metadatas = get_metadata(root=path)

        methods = set(methods) if methods is not None else None
        metadatas = {
            k: v
            for k, v in metadatas.items()
            if (max_seed is None or v["seed"] < max_seed)
               and (methods is None or v["algorithm"] in methods)
        }
        # Save original algorithms before they might get renamed
        for k, v in metadatas.items():
            v['original_algorithm'] = v['algorithm']
            
        if experiment_filter:
            metadatas = {k: v for k, v in metadatas.items() if experiment_filter(v)}
        logger.info(f"Loaded {len(metadatas)} experiment metadata items matching criteria.")
        # metadatas = {k: v for k, v in metadatas.items() if "yahpo" not in v["benchmark"]}

        # Snapshot taken before --only_best/--rename_best may reassign/filter `metadatas` below,
        # so --emit_advantage_weighted always runs over the full pool independent of that pipeline.
        all_metadatas_for_advantage = dict(metadatas)

        true_metric_range_by_benchmark = None
        if args.metric_range_mode == "true":
            with catchtime("Compute true per-benchmark metric ranges"):
                true_metric_range_by_benchmark = compute_true_metric_range_per_benchmark(
                    metadatas, path
                )

        if args.only_best:
            with catchtime("Find best trajectories for each benchmark"):
                best_by_benchmark = find_best_algorithms(metadatas, path)

                if args.keep_all_seeds_of_best:
                    best_algorithms = {
                        bench: r["algorithm"] for bench, r in best_by_benchmark.items()
                    }
                    new_metadatas = {}
                    for k, v in metadatas.items():
                        bench = v.get('benchmark', v.get('entrypoint', 'unknown'))
                        if bench in best_algorithms and v["algorithm"] == best_algorithms[bench]:
                            new_metadatas[k] = v
                    metadatas = new_metadatas
                else:
                    best_names = {
                        r["best_experiment"][0] for r in best_by_benchmark.values()
                        if r["best_experiment"] is not None
                    }
                    metadatas = {k: v for k, v in metadatas.items() if k in best_names}

                if args.rename_best:
                    for v in metadatas.values():
                        v['algorithm'] = 'best'
                        
                logger.info(f"Filtered down to {len(metadatas)} best experiment metadata entries.")

        summary_data = []
        for k, v in metadatas.items():
            summary_data.append({
                "experiment_name": k,
                "benchmark": v.get("benchmark", v.get("entrypoint", "unknown")),
                # If renamed, we try to recover the original algorithm if needed, 
                # but wait, if it's renamed, `v['algorithm']` is already 'best'.
                # Let's just output the current algorithm. Wait, the user wants the distribution!
                "algorithm": v.get("original_algorithm", v["algorithm"])
            })
        with open(str(output_path / "dataset_summary.json"), "w") as f:
            json.dump(summary_data, f, indent=4)

        with catchtime("Load results dataframes"):
            # load results in parallel

            hist_train = list()
            hist_valid = list()
            
            perm_config = {}
            if args.permutation_config and os.path.exists(args.permutation_config):
                with open(args.permutation_config, 'r') as f:
                    perm_config = json.load(f)

            def get_dataset_category(metadata):
                b_name = metadata.get('benchmark', metadata.get('entrypoint', 'unknown'))
                masked = len(metadata.get('masked_params', [])) > 0
                
                if b_name.startswith('fcnet'):
                    return 'Masked FC-Net' if masked else 'FC-Net'
                elif b_name.startswith('nas201'):
                    return 'Masked NAS-Bench-201' if masked else 'NAS-Bench-201'
                elif b_name.startswith('lcbench'):
                    return 'LC-Bench'
                elif b_name.startswith('pd1'):
                    return 'PD1'
                elif b_name.startswith('hpob'):
                    return 'HPO-B'
                elif b_name.startswith('tabrepo'):
                    return 'TabRepo'
                elif b_name.startswith('global-optimization'):
                    return 'Global Optimization'
                else:
                    return 'Unknown'
            
            tasks_metadata = []
            for name, metadata in metadatas.items():
                benchmark_name = metadata.get('benchmark', '')
                is_valid = benchmark_name in validation_tasks
                
                if perm_config:
                    category = get_dataset_category(metadata)
                    num_perm = perm_config.get(category, args.num_permutation)
                else:
                    num_perm = args.num_permutation
                
                tasks_metadata.append((name, metadata, path, max_num_trials, args.remove_names, num_perm, args.sample_shorter_trajectories, is_valid, true_metric_range_by_benchmark))
            
            try:
                num_cores = len(os.sched_getaffinity(0))
            except AttributeError:
                num_cores = multiprocessing.cpu_count()
                
            logger.info(f"Loading and processing dataframes using {num_cores} cores for {len(tasks_metadata)} tasks...")
                
            with multiprocessing.Pool(processes=num_cores) as pool:
                for is_valid, histories in tqdm.tqdm(
                        pool.imap_unordered(process_metadata, tasks_metadata), 
                        total=len(tasks_metadata),
                        desc="Loading results",
                        mininterval=5.0):
                    if is_valid:
                        hist_valid.extend(histories)
                    else:
                        hist_train.extend(histories)

            logger.info(f"Data loading complete. Writing outputs to {output_path}...")
            random.shuffle(hist_train)
            for split in ['train', 'valid']:
                file_name = f"{split}.txt"
                if split == 'train':
                    hist_split = hist_train
                else:
                    hist_split = hist_valid
                with open(str(output_path / file_name), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(hist_split))

            del hist_train, hist_valid

        if args.emit_advantage_weighted:
            with catchtime("Compute advantage-weighted training data"):
                peer_baseline_by_benchmark = compute_peer_baseline_per_benchmark(
                    all_metadatas_for_advantage, path
                )

                adv_tasks = []
                for name, metadata in all_metadatas_for_advantage.items():
                    benchmark_name = metadata.get('benchmark', '')
                    is_valid = benchmark_name in validation_tasks
                    adv_tasks.append(
                        (name, metadata, path, peer_baseline_by_benchmark, args.advantage_temperature, is_valid,
                         true_metric_range_by_benchmark)
                    )

                logger.info(
                    f"Computing advantage-weighted examples using {num_cores} cores for {len(adv_tasks)} tasks..."
                )

                adv_train = []
                adv_valid = []
                with multiprocessing.Pool(processes=num_cores) as pool:
                    for is_valid, example in tqdm.tqdm(
                            pool.imap_unordered(process_metadata_with_advantage, adv_tasks),
                            total=len(adv_tasks),
                            desc="Computing advantage-weighted examples",
                            mininterval=5.0):
                        if example is None:
                            continue
                        if is_valid:
                            adv_valid.append(example)
                        else:
                            adv_train.append(example)

                random.shuffle(adv_train)
                for split, examples in [('train', adv_train), ('valid', adv_valid)]:
                    file_name = f"advantage_{split}.jsonl"
                    with open(str(output_path / file_name), 'w', encoding='utf-8') as f:
                        for example in examples:
                            f.write(json.dumps(example) + '\n')

                logger.info(
                    f"Wrote {len(adv_train)} advantage-weighted train examples and "
                    f"{len(adv_valid)} valid examples to {output_path}."
                )
