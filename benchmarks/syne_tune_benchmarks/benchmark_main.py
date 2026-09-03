import itertools
import logging
import gc
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner

from baselines import (
    MethodArguments,
    methods,
)


def run(
    method_names,
    benchmark_names,
    seeds,
    checkpoint_dir,
    max_num_evaluations=None,
    n_workers: int = 1,
    gpu_memory_utilization: float = 0.2,
):
    logging.getLogger("syne_tune.optimizer.schedulers").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend.simulator_backend.simulator_backend").setLevel(
        logging.WARNING
    )

    combinations = list(itertools.product(method_names, seeds, benchmark_names))

    optformer_model, optformer_tokenizer = None, None
    if checkpoint_dir and any(m.upper().startswith("OPT") for m in method_names):
        from open_optformer.optformer_searcher import load_optformer_model

        optformer_model, optformer_tokenizer, _ = load_optformer_model(
            Path(checkpoint_dir),
            gpu_memory_utilization=gpu_memory_utilization,
            use_vllm=True,
        )

    print(f"Going to evaluate: {combinations}")
    exp_names = []
    for method, seed, benchmark_name in tqdm(combinations):
        np.random.seed(seed)
        if benchmark_name.startswith("hpob"):
            from hpob_benchmarks import hpob_benchmark_definitions
            benchmark = hpob_benchmark_definitions[benchmark_name]
        elif benchmark_name.startswith("tabrepo"):
            from tabrepo_benchmarks import tabrepo_benchmark_definitions
            benchmark = tabrepo_benchmark_definitions[benchmark_name]
        elif benchmark_name.startswith("pd1"):
            from pd1_benchmarks import pd1_benchmark_definitions
            benchmark = pd1_benchmark_definitions[benchmark_name]
        elif benchmark_name.startswith("deepar"):
            from deepar_benchmarks import deepar_benchmark_definitions
            benchmark = deepar_benchmark_definitions[benchmark_name]
        elif benchmark_name.startswith("fcnet"):
            from fcnet_benchmarks import fcnet_benchmark_definitions
            benchmark = fcnet_benchmark_definitions[benchmark_name]
        elif benchmark_name.startswith("nas201"):
            from nas201_benchmarks import nas201_benchmark_definitions
            benchmark = nas201_benchmark_definitions[benchmark_name]
        elif benchmark_name.startswith("lcbench"):
            from lcbench_benchmarks import lcbench_benchmark_definitions
            benchmark = lcbench_benchmark_definitions[benchmark_name]
        elif benchmark_name.startswith("bbomix"):
            from bbomix_benchmarks import bbomix_benchmark_definitions
            benchmark = bbomix_benchmark_definitions[benchmark_name]
        else:
            raise NotImplementedError(f"Unknown benchmark name: {benchmark_name}")
            
        print(f"Starting experiment ({method}/{benchmark_name}/{seed})")

        backend = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
            surrogate=getattr(benchmark, "surrogate", None),
            surrogate_kwargs=getattr(benchmark, "surrogate_kwargs", None),
        )

        # 5 candidates initially to be evaluated
        num_random_candidates = 5
        random_state = np.random.RandomState(seed)
        points_to_evaluate = [
            {
                k: v.sample(random_state=random_state) if hasattr(v, "sample") else v
                for k, v in backend.blackbox.configuration_space.items()
            }
            for _ in range(num_random_candidates)
        ]
        
        scheduler = methods[method](
            MethodArguments(
                benchmark_name=benchmark.blackbox_name + "_" + benchmark.dataset_name,
                config_space=backend.blackbox.configuration_space,
                metric=benchmark.metric,
                mode=benchmark.mode,
                random_seed=seed,
                num_brackets=1,
                checkpoint_dir=checkpoint_dir if checkpoint_dir else None,
                use_surrogates=getattr(benchmark, "use_surrogate", False),
                points_to_evaluate=points_to_evaluate,
                model=optformer_model,
                tokenizer=optformer_tokenizer,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        )

        stop_criterion = StoppingCriterion(
            max_num_trials_completed=max_num_evaluations
            if max_num_evaluations
            else benchmark.max_num_evaluations,
        )

        # Safely derive output directory paths
        if checkpoint_dir:
            ckpt_path = Path(checkpoint_dir).resolve()
            model_name = ckpt_path.name
            variant_name = ckpt_path.parent.name
        else:
            model_name = "default_model"
            variant_name = "default_variant"

        suffix = f"{method}-{seed}-{benchmark_name}".replace("_", "-")
        tuner_dir = f"/data/horse/ws/luth474h-bbo-pile-experiments/masters-thesis/results/{model_name}/{variant_name}/{suffix}"

        tuner = Tuner(
            trial_backend=backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=n_workers,  # Passed n_workers properly
            sleep_time=0,
            callbacks=[SimulatorCallback()],
            results_update_interval=600,
            print_update_interval=30,
            tuner_name=tuner_dir,
            save_tuner=False,
            suffix_tuner_name=False,
            metadata={
                "seed": seed,
                "algorithm": method,
                "benchmark": benchmark_name,
            },
        )
        tuner.run()
        exp_names.append(tuner.name)

        # Cleanup references directly in caller frame
        del tuner
        del scheduler
        del backend
        del points_to_evaluate
        gc.collect()

    return exp_names


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0,
        help="seed to run",
    )
    parser.add_argument(
        "--run_all_seeds",
        type=int,
        required=False,
        default=0,
        help="If 1 runs all seeds between [0, args.seed] if 0 run only args.seed.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=False,
        help="a method to run from baselines.py, run all by default.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="a benchmark to run",
    )
    parser.add_argument(
        "--n_workers",
        help="number of workers to use when tuning.",
        type=int,
        default=1,
    )
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
        seeds = list(range(args.seed + 1))
    else:
        seeds = [args.seed]

    if args.method is None or args.method.upper().startswith("OptFormer"):
#        # avoid importing google vizier dependencies if not needed
        from original_optformer_methods import original_optformer_methods
        methods = original_optformer_methods | methods

    method_names = [args.method] if args.method is not None else list(methods.keys())

    run(
        method_names=method_names,
        checkpoint_dir=args.checkpoint_dir,
        benchmark_names=[args.benchmark],
        seeds=seeds,
        n_workers=args.n_workers,
        gpu_memory_utilization=args.gpu_utilization,
    )

    os._exit(0)
