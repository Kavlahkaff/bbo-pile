from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)

from open_optformer.optformer_searcher import OptformerScheduler

@dataclass
class MethodArguments:
    config_space: dict
    metric: str
    mode: str
    random_seed: int
    resource_attr: str
    points_to_evaluate: List[dict]
    max_t: Optional[int] = None
    max_resource_attr: Optional[str] = None
    use_surrogates: bool = False
    num_brackets: Optional[int] = 1
    verbose: Optional[bool] = False
    checkpoint_dir: Optional[str] = None
    benchmark_name: Optional[str] = None


class Methods:
    # single fidelity
    BORE = "BORE"
    RS = "RS"
    TPE = "TPE"
    REA = "REA"
    BOTorch = "BOTorch"
    CQR = "CQR"
    OptFormerBBOB_HillClimb = "OptFormerBBOB-HillClimb"
    OptFormerBBOB_GP = "OptFormerBBOB-GP"
    OptFormerBBOB_REGEVO = "OptFormerBBOB-REGEVO"
    OptFormerBBOB_RS = "OptFormerBBOB-RS"
    OptFormerHPOB_GP = "OptFormerHPOB-GP"
    OPT_RS = "OPT-RS"
    OPT_REA = "OPT-REA"

methods = {
    Methods.RS: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="random_search",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.BORE: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="bore",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.TPE: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="kde",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate, 'min_bandwidth': 1e-1},
    ),
    Methods.CQR: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="cqr",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.BOTorch: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="botorch",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.REA: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="regularized_evolution",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.OPT_RS: lambda method_arguments: OptformerScheduler(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        checkpoint_dir=Path(method_arguments.checkpoint_dir),
        task_info = {'name': method_arguments.benchmark_name,
                    'algorithm': "RS",
                    'metric_names': "feval"},
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
    ),

    Methods.OPT_REA: lambda method_arguments: OptformerScheduler(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        checkpoint_dir=Path(method_arguments.checkpoint_dir),
        task_info={'name': method_arguments.benchmark_name,
                   'algorithm': "REA",
                   'metric_names': "feval"},
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
    ),
}


if __name__ == "__main__":
    # Run a loop that initializes all schedulers on all benchmark to see if they all work
    from benchmarks import (
        benchmark_definitions,
    )

    print(f"Checking initialization of {list(methods.keys())[::-1]}")

    benchmarks = [
        "fcnet-protein",
        "nas201-cifar10",
        "lcbench-Fashion-MNIST",
        "tabrepo-RandomForest-2dplanes",
        "hpob_5636_3492",
    ]
    for benchmark_name in benchmarks:
        benchmark = benchmark_definitions[benchmark_name]
        backend = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
        )
        points_to_evaluate = [
            {
                k: v.sample() if hasattr(v, "sample") else v
                for k, v in backend.blackbox.configuration_space.items()
            }
            for _ in range(4)
        ]
        print(f"Checking initialization of {list(methods.keys())[::-1]}")
        for method_name, method_fun in list(methods.items())[::-1]:
            print(f"checking initialization of: {method_name}, {benchmark_name}")
            # if method_name != Methods.QHB_XGB:
            #     continue

            scheduler = method_fun(
                MethodArguments(
                    config_space=backend.blackbox.configuration_space,
                    metric=benchmark.metric,
                    mode=benchmark.mode,
                    random_seed=0,
                    max_t=max(backend.blackbox.fidelity_values),
                    resource_attr=next(iter(backend.blackbox.fidelity_space.keys())),
                    use_surrogates=benchmark_name == "lcbench-Fashion-MNIST",
                    points_to_evaluate=points_to_evaluate,
                )
            )
            if isinstance(scheduler, TrialScheduler):
                print(scheduler.suggest())
                print(scheduler.suggest())
            else:
                print(scheduler.suggest(0))
                print(scheduler.suggest(1))
