from dataclasses import dataclass

from syne_tune.optimizer.baselines import REA, RandomSearch, CQR, TPE, BOTorch, BORE
from syne_tune.optimizer.schedulers.smac_scheduler import SMACScheduler
from syne_tune.optimizer.schedulers.single_objective_scheduler import SingleObjectiveScheduler
from open_optformer.hebo_searcher import HEBOSearcher

@dataclass
class MethodArguments:
    config_space: dict
    metric: str
    mode: str
    random_seed: int
    points_to_evaluate: list[dict]


class Methods:
    BORE = "BORE"
    RS = "RS"
    TPE = "TPE"
    REA = "REA"
    BOTorch = "BOTorch"
    CQR = "CQR"
    HEBO = 'HEBO'
    SMAC = 'SMAC'

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
    Methods.SMAC: lambda method_arguments: SMACScheduler(
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
}