import json
import pandas as pd
import os
import concurrent.futures
import logging
from tqdm import tqdm

from pathlib import Path
from pyparfor import parfor

from json import JSONDecodeError

logger = logging.getLogger(__name__)

from syne_tune.experiments import ExperimentResult
from syne_tune.config_space import config_space_from_json_dict
from open_optformer.history import History


def get_config_space_from_metadata(metadata):
    if 'config_space' in metadata:
        return config_space_from_json_dict(json.loads(metadata['config_space']))
    
    benchmark_name = metadata.get('benchmark', '')
    if benchmark_name.startswith('global-optimization'):
        import sys
        from syne_tune.config_space import config_space_to_json_dict
        
        # Dynamically include the benchmark directory in sys.path
        import_path = str(Path(__file__).parent.parent / "benchmarks" / "global_optimization_problems")
        if import_path not in sys.path:
            sys.path.append(import_path)
            
        from benchmarks_definitions import benchmark_definitions
        
        if benchmark_name in benchmark_definitions:
            config_space = benchmark_definitions[benchmark_name].configuration_space
            # Artificially inject it back into metadata so history generation can see it
            metadata['config_space'] = json.dumps(config_space_to_json_dict(config_space))
            return config_space
            
    raise KeyError(f"'config_space' missing in metadata for benchmark '{benchmark_name}'")


def load_result(name, metric_name, config_space, path):
    usecols = [metric_name, "st_tuner_time", "trial_id", "st_decision"]
    usecols.extend(['config_{}'.format(k) for k in config_space.keys()])
    try:
        return pd.read_csv(path / name / "results.csv.zip", usecols=usecols)
    except Exception:
        return None


def create_history_from_results(name, metadata, path: Path,
                                max_num_trials: int,
                                num_numeric_tokens: int = 1000,
                                remove_names: bool = False,
                                n_permutation: int = 0) -> list[str]:
    config_space = get_config_space_from_metadata(metadata)
    metric_name = metadata["metric_names"][0]
    res = load_result(name, metric_name, config_space, path)

    hist = History.from_syne_tune_experiment(ExperimentResult(name=name,
                                                              metadata=metadata,
                                                              results=res,
                                                              path=path,
                                                              tuner=None),
                                             num_numeric_tokens=num_numeric_tokens,
                                             remove_names=remove_names,
                                             max_num_trials=max_num_trials)
    traj = list()
    traj.append(hist.get_prompt())
    for i in range(n_permutation):
        traj.append(hist.get_prompt(shuffle=True))
    return traj

def read_single_metadata(args):
    metadata_path, root_path = args
    try:
        with open(metadata_path, "r") as f:
            rel_path = str(Path(metadata_path).parent.relative_to(root_path))
            return rel_path, json.load(f)
    except JSONDecodeError as e:
        logger.error(f"JSONDecodeError at {metadata_path}")
        raise e

def get_metadata(root: Path):
    metadatas = {}
    metadata_paths = []
    
    logger.info(f"Scanning directory tree at '{root}' for metadata files (using os.walk)...")
    for dirpath, dirnames, filenames in os.walk(str(root)):
        for filename in filenames:
            if filename.endswith("metadata.json"):
                metadata_paths.append((os.path.join(dirpath, filename), root))
                
    logger.info(f"Found {len(metadata_paths)} metadata files. Loading in parallel...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(
            executor.map(read_single_metadata, metadata_paths), 
            total=len(metadata_paths), 
            desc="Reading metadata JSONs", 
            mininterval=5.0
        ))
        
    for rel_path, data in results:
        metadatas[rel_path] = data

    return metadatas
