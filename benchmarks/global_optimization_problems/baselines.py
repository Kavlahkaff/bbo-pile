from dataclasses import dataclass
from pathlib import Path
from syne_tune.optimizer.baselines import REA, RandomSearch, CQR, TPE, BOTorch, BORE
from syne_tune.optimizer.schedulers.single_objective_scheduler import SingleObjectiveScheduler
from open_optformer.hebo_searcher import HEBOSearcher
from open_optformer.optformer_searcher import OptformerScheduler

@dataclass
class MethodArguments:
    config_space: dict
    metric: str
    mode: str
    random_seed: int
    points_to_evaluate: list[dict]
    checkpoint_dir: str
    benchmark_name: str

class Methods:
    BORE = "BORE"
    RS = "RS"
    TPE = "TPE"
    REA = "REA"
    BOTorch = "BOTorch"
    CQR = "CQR"
    HEBO = 'HEBO'
    OPT_CQR = 'OPT-CQR'
    OPT_REA = 'OPT-REA'
    OPT_BORE = 'OPT-BORE'
    OPT_TPE = 'OPT-TPE'
    OPT_HEBO = 'OPT-HEBO'
    OPT_CQR_TS = 'OPT-CQR-TS'
    OPT_CQR_TS_5 = 'OPT-CQR-TS-5'

methods = {
    Methods.RS: lambda method_arguments: RandomSearch(
        config_space=method_arguments.config_space,
        metrics=[method_arguments.metric],
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate
    ),
    Methods.BORE: lambda method_arguments: BORE(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate
    ),
    Methods.CQR: lambda method_arguments: CQR(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate
    ),
    Methods.TPE: lambda method_arguments: TPE(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate
    ),
    Methods.REA: lambda method_arguments: REA(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate
    ),
    Methods.BOTorch: lambda method_arguments: BOTorch(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate
    ),
    Methods.HEBO: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher=HEBOSearcher(
            config_space=method_arguments.config_space,
            do_minimize=method_arguments.mode == "min",
            random_seed=method_arguments.random_seed,
            points_to_evaluate=method_arguments.points_to_evaluate
        ),
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
    ),
    Methods.OPT_CQR: lambda method_arguments: OptformerScheduler(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        checkpoint_dir=Path(method_arguments.checkpoint_dir),
        task_info={'name': method_arguments.benchmark_name,
                   'algorithm': "CQR",
                   'metric_names': "feval"},
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
        n_sample_configurations=1,
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
        n_sample_configurations=1,
    ),
    Methods.OPT_BORE: lambda method_arguments: OptformerScheduler(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        checkpoint_dir=Path(method_arguments.checkpoint_dir),
        task_info={'name': method_arguments.benchmark_name,
                   'algorithm': "BORE",
                   'metric_names': "feval"},
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
        n_sample_configurations=1,
    ),
    Methods.OPT_TPE: lambda method_arguments: OptformerScheduler(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        checkpoint_dir=Path(method_arguments.checkpoint_dir),
        task_info={'name': method_arguments.benchmark_name,
                   'algorithm': "TPE",
                   'metric_names': "feval"},
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
        n_sample_configurations=1,
    ),
    Methods.OPT_HEBO: lambda method_arguments: OptformerScheduler(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        checkpoint_dir=Path(method_arguments.checkpoint_dir),
        task_info={'name': method_arguments.benchmark_name,
                   'algorithm': "HEBO",
                   'metric_names': "feval"},
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
        n_sample_configurations=1,
    ),
    Methods.OPT_CQR_TS: lambda method_arguments: OptformerScheduler(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        checkpoint_dir=Path(method_arguments.checkpoint_dir),
        task_info={'name': method_arguments.benchmark_name,
                   'algorithm': "CQR",
                   'metric_names': "feval"},
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
        n_sample_configurations=50,
    ),
    Methods.OPT_CQR_TS_5: lambda method_arguments: OptformerScheduler(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        checkpoint_dir=Path(method_arguments.checkpoint_dir),
        task_info={'name': method_arguments.benchmark_name,
                   'algorithm': "CQR",
                   'metric_names': "feval"},
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
        n_sample_configurations=5,
    ),
}
