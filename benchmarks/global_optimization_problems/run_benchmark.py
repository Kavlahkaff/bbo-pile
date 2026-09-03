import json
import os
import time

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import pandas

from syne_tune.optimizer.schedulers.ask_tell_scheduler import AskTellScheduler
from syne_tune.tuning_status import Status
from syne_tune.config_space import config_space_to_json_dict
from baselines import methods, MethodArguments
from benchmarks_definitions import benchmark_definitions

if __name__ == "__main__":
    import logging
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run_all_seeds", type=int, required=False, default=0)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--max_trials", type=int, default=100)
    parser.add_argument("--output_path", type=str, default='/data/horse/ws/luth474h-bbo-pile-experiments/masters-thesis/results_global')
    parser.add_argument("--num_start_points", type=int, default=5)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=False,
        default="",
        help="directory for optformer model checkpoints",
    )
    parser.add_argument(
        "--gpu_utilization",
        type=float,
        required=False,
        default=0.2,
        help="GPU memory utilization for vLLM",
    )
    args, _ = parser.parse_known_args()

    if args.run_all_seeds:
        seeds = list(range(args.seed))
    else:
        seeds = [args.seed]

    method = args.method
    benchmark = args.benchmark
    max_trials = args.max_trials
    metric = "y"
    mode = "min"

    optformer_model, optformer_tokenizer = None, None
    if args.checkpoint_dir and method.startswith("OPT"):
        from open_optformer.optformer_searcher import load_optformer_model
        optformer_model, optformer_tokenizer, _ = load_optformer_model(
            Path(args.checkpoint_dir), 
            gpu_memory_utilization=args.gpu_utilization, 
            use_vllm=True
        )
    # Safely derive output directory paths
    if args.checkpoint_dir:
        ckpt_path = Path(args.checkpoint_dir).resolve()
        model_name = ckpt_path.name
        variant_name = ckpt_path.parent.name
    else:
        model_name = "default_model"
        variant_name = "default_variant"

    base_dir = f"/data/horse/ws/luth474h-bbo-pile-experiments/masters-thesis/results_global/{model_name}/{variant_name}"
    for seed in seeds:
        tuner_dir = Path(base_dir) / f'{benchmark}_{method}_seed_{seed}'
        os.makedirs(tuner_dir, exist_ok=True)
        root = logging.getLogger()
        root.setLevel(logging.INFO)

        blackbox = benchmark_definitions[benchmark]
        config_space = blackbox.configuration_space
        points_to_evaluate = [{k: v.sample() for k, v in config_space.items()} for _ in range(args.num_start_points)]
        scheduler = methods[method](MethodArguments(
                metric=metric,
                random_seed=seed,
                mode='min',
                config_space=config_space,
                points_to_evaluate=points_to_evaluate,
                checkpoint_dir=args.checkpoint_dir,
                benchmark_name=args.benchmark,
                model=optformer_model,
                tokenizer=optformer_tokenizer,
                gpu_memory_utilization=args.gpu_utilization
            )
        )

        scheduler = AskTellScheduler(
            base_scheduler=scheduler,
        )

        start_time = time.time()

        results_dict = defaultdict(list)

        for iter in range(max_trials):
            trial_suggestion = scheduler.ask()
            result = blackbox(trial_suggestion.config)
            scheduler.tell(trial_suggestion, result)
            runtime = time.time() - start_time
            print(f'iteration: {iter}, evaluated x={trial_suggestion.config}, objective={result[metric]}, runtime={runtime}')

            for hp_name, hp_value in trial_suggestion.config.items():
                results_dict[f'config_{hp_name}'].append(hp_value)
            results_dict['objective'].append(result[metric])
            results_dict['st_tuner_time'].append(runtime)
            results_dict['st_decision'].append(Status.completed)
            results_dict['trial_id'].append(trial_suggestion.trial_id)


        results = pandas.DataFrame(results_dict)
        results.to_csv(tuner_dir / 'results.csv.zip',  compression={'method': 'zip'})
        metadata = {"algorithm": method,
                    "benchmark": 'global-optimization_' + benchmark,
                    "seed": seed,
                    "config_space":  json.dumps(config_space_to_json_dict(config_space)),
                    'metric_names': ['objective']}

        json.dump(metadata, open(tuner_dir / 'metadata.json', 'w'))

        del scheduler
        del blackbox
        del points_to_evaluate
        del results_dict
        del results
        del metadata

    if optformer_model is not None:
        del optformer_model
        del optformer_tokenizer
    if args.checkpoint_dir and method.startswith("OPT"):
        # Force kill lingering vLLM/Ray daemons to release cluster allocation
        os._exit(0)
