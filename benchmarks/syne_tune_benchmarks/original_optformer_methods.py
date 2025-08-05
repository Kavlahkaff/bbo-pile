from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)

from open_optformer.original_optformer_searcher import OriginalOptFormerSearcher
from baselines import Methods

original_optformer_methods = {

    Methods.OptFormerHillClimb:  lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher=OriginalOptFormerSearcher(points_to_evaluate=method_arguments.points_to_evaluate,
                                   config_space=method_arguments.config_space,
                                   random_seed=method_arguments.random_seed,
                                           designer_name='designer_hill_climb',
                                   ),
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
    ),
    Methods.OptFormerGPUCB: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher=OriginalOptFormerSearcher(points_to_evaluate=method_arguments.points_to_evaluate,
                                           config_space=method_arguments.config_space,
                                           random_seed=method_arguments.random_seed,
                                           designer_name='designer_recursive_gp',
                                           ),
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
    ),
    Methods.OptFormerRS: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher=OriginalOptFormerSearcher(points_to_evaluate=method_arguments.points_to_evaluate,
                                           config_space=method_arguments.config_space,
                                           random_seed=method_arguments.random_seed,
                                           designer_name='designer_random_search',
                                           ),
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
    ),
}