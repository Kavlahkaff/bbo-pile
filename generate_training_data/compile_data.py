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
    name, metadata, path, use_auc = args_tuple
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
        if use_auc:
            if metric_mode == 'max':
                val = -np.maximum.accumulate(res[metric_name]).sum()
            else:
                val = np.minimum.accumulate(res[metric_name]).sum()
        else:
            if metric_mode == 'max':
                val = -res[metric_name].max()
            else:
                val = res[metric_name].min()
    return benchmark_name, name, val

def process_metadata(args_tuple):
    name, metadata, path, max_num_trials, remove_names, num_permutation, sample_shorter_trajectories, is_valid = args_tuple
    from load_data import create_history_from_results
    histories = []
    
    try:
        histories.extend(create_history_from_results(name, metadata, path, max_num_trials,
                                                     remove_names=remove_names,
                                                     n_permutation=num_permutation))
        if not is_valid and sample_shorter_trajectories:
            for mt in [1, 5, 10, 20]:
                histories.extend(create_history_from_results(name, metadata, path,
                                                             mt,
                                                             remove_names=remove_names,
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
        "--best_by_auc",
        action='store_true',
        help="when using --only_best, select the best trajectory based on Area Under the Curve (AUC) of the cumulative best instead of the final best value",
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

        if args.only_best:
            with catchtime("Find best trajectories for each benchmark"):
                best_experiments = {}
                tasks_best = [(name, metadata, path, args.best_by_auc) for name, metadata in metadatas.items()]
                
                try:
                    num_cores = len(os.sched_getaffinity(0))
                except AttributeError:
                    num_cores = multiprocessing.cpu_count()
                
                logger.info(f"Detecting best trajectories using {num_cores} cores for {len(tasks_best)} tasks...")
                
                with multiprocessing.Pool(processes=num_cores) as pool:
                    for benchmark_name, name, val in tqdm.tqdm(
                            pool.imap_unordered(process_best_trajectory, tasks_best), 
                            total=len(tasks_best), 
                            desc="Finding best trajectories",
                            mininterval=5.0):
                        if val is not None:
                            if benchmark_name not in best_experiments or val < best_experiments[benchmark_name][1]:
                                best_experiments[benchmark_name] = (name, val)

                if args.keep_all_seeds_of_best:
                    best_algorithms = {
                        bench: metadatas[name]["algorithm"] 
                        for bench, (name, _) in best_experiments.items()
                    }
                    new_metadatas = {}
                    for k, v in metadatas.items():
                        bench = v.get('benchmark', v.get('entrypoint', 'unknown'))
                        if bench in best_algorithms and v["algorithm"] == best_algorithms[bench]:
                            new_metadatas[k] = v
                    metadatas = new_metadatas
                else:
                    best_names = {v[0] for v in best_experiments.values()}
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
            
            tasks_metadata = []
            for name, metadata in metadatas.items():
                benchmark_name = metadata.get('benchmark', '')
                is_valid = benchmark_name in validation_tasks
                tasks_metadata.append((name, metadata, path, max_num_trials, args.remove_names, args.num_permutation, args.sample_shorter_trajectories, is_valid))
            
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