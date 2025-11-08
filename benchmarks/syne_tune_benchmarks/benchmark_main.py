import itertools
import logging
from argparse import ArgumentParser

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
from fcnet_benchmarks import benchmark_definitions

def run(
    method_names,
    benchmark_names,
    seeds,
    checkpoint_dir,
    max_num_evaluations=None,
    n_workers: int = 4,
):
    logging.getLogger("syne_tune.optimizer.schedulers").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend.simulator_backend.simulator_backend").setLevel(
        logging.WARNING
    )

    combinations = list(itertools.product(method_names, seeds, benchmark_names))

    print(f"Going to evaluate: {combinations}")
    exp_names = []
    for method, seed, benchmark_name in tqdm(combinations):
        np.random.seed(seed)
        if benchmark_name.startswith("hpob_"):
            from hpob_benchmarks import hpob_benchmark_definitions
            benchmark = hpob_benchmark_definitions[benchmark_name]
        elif benchmark_name.startswith("tabrepo_"):
            from tabrepo_benchmarks import tabrepo_benchmark_definitions
            benchmark = tabrepo_benchmark_definitions[benchmark_name]
        elif benchmark_name.startswith("pd1_"):
            from pd1_benchmarks import pd1_benchmark_definitions
            benchmark = pd1_benchmark_definitions[benchmark_name]
        elif benchmark_name.startswith("deepar_"):
            from deepar_benchmarks import deepar_benchmark_definitions
            benchmark = deepar_benchmark_definitions[benchmark_name]
        else:
            benchmark = benchmark_definitions[benchmark_name]

        print(f"Starting experiment ({method}/{benchmark_name}/{seed})")

        backend = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
            surrogate=benchmark.surrogate,
            surrogate_kwargs=benchmark.surrogate_kwargs,
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
                benchmark_name=benchmark.blackbox_name + '_' + benchmark.dataset_name,
                config_space=backend.blackbox.configuration_space,
                metric=benchmark.metric,
                mode=benchmark.mode,
                random_seed=seed,
                num_brackets=1,
                checkpoint_dir=checkpoint_dir,
                use_surrogates=benchmark.use_surrogate,
                points_to_evaluate=points_to_evaluate,

            )
        )

        stop_criterion = StoppingCriterion(
#            max_wallclock_time=benchmark.max_wallclock_time,
            max_num_trials_completed=max_num_evaluations
            if max_num_evaluations
            else benchmark.max_num_evaluations,
        )
        tuner = Tuner(
            trial_backend=backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=n_workers,
            sleep_time=0,
            callbacks=[SimulatorCallback()],
            results_update_interval=600,
            print_update_interval=30,
            # we set a convenient name for tuner to retrieve results easily
            tuner_name=f"results/{method}-{seed}-{benchmark_name}".replace("_", "-"),
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
        required=False,
        help="a benchmark to run from blackbox_benchmarks.py, run all by default.",
    )
    parser.add_argument(
        "--benchmark_family",
        type=str,
        required=False,
        help="run all benchmarks of the specified family only (e.g., 'hpob', 'tabrepo')",
    )
    parser.add_argument(
        "--n_workers",
        help="number of workers to use when tuning.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=False,
        default="",
        help="directory for optformer model checkpoints",
    )
    args, _ = parser.parse_known_args()
    if args.run_all_seeds:
        seeds = list(range(args.seed))
    else:
        seeds = [args.seed]
        
    if args.method is None or args.method.startswith("OptFormer"):
        # avoid importing nasty google vizier dependencies if we don't need them
        from original_optformer_methods import original_optformer_methods
        methods  = original_optformer_methods | methods
    method_names = [args.method] if args.method is not None else list(methods.keys())

    if args.benchmark is not None:
       benchmark_names = [args.benchmark]

    else:
        # this avoids downloading all benchmarks
        if args.benchmark_family == 'hpob':
            from hpob_benchmarks import hpob_benchmark_definitions
            benchmark_names = list(hpob_benchmark_definitions.keys())
        elif args.benchmark_family == 'tabrepo':
            from tabrepo_benchmarks import tabrepo_benchmark_definitions
            benchmark_names = list(tabrepo_benchmark_definitions.keys())
        elif args.benchmark_family == 'pd1':
            from pd1_benchmarks import pd1_benchmark_definitions
            benchmark_names = list(pd1_benchmark_definitions.keys())
        elif args.benchmark_family == 'deepar':
            from deepar_benchmarks import deepar_benchmark_definitions
            benchmark_names = list(deepar_benchmark_definitions.keys())
        elif args.benchmark_family == 'fcnet':
            from fcnet_benchmarks import fcnet_benchmark_definitions
            benchmark_names = list(fcnet_benchmark_definitions.keys())
        elif args.benchmark_family == 'nas201':
            from nas201_benchmarks import nas201_benchmark_definitions
            benchmark_names = list(nas201_benchmark_definitions.keys())
        elif args.benchmark_family == 'lcbench':
            from lcbench_benchmarks import lcbench_benchmark_definitions
            benchmark_names = list(lcbench_benchmark_definitions.keys())

        benchmark_names =  list(benchmark_definitions.keys())
    run(
        method_names=method_names,
        checkpoint_dir=args.checkpoint_dir,
        benchmark_names=benchmark_names,
        seeds=seeds,
        n_workers=args.n_workers,
    )
