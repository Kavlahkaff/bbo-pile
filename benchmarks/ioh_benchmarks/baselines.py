from dataclasses import dataclass
from pathlib import Path
from syne_tune.optimizer.baselines import REA, RandomSearch, CQR, TPE, BORE
from syne_tune.optimizer.schedulers.single_objective_scheduler import SingleObjectiveScheduler
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
    RS = "RS"
    BORE = "BORE"
    CQR = "CQR"
    TPE = "TPE"
    REA = "REA"
    #OPT_CQR = "OPT-CQR"
    #OPT_REA = "OPT-REA"
    #OPT_BORE = "OPT-BORE"
    #OPT_TPE = "OPT-TPE"
    #OPT_CQR_TS = "OPT-CQR-TS"
    #OPT_CQR_TS_5 = "OPT-CQR-TS-5"

methods = {
    Methods.RS: lambda ma: RandomSearch(config_space=ma.config_space, metrics=[ma.metric], do_minimize=ma.mode=="min", random_seed=ma.random_seed, points_to_evaluate=ma.points_to_evaluate),
    Methods.BORE: lambda ma: BORE(config_space=ma.config_space, metric=ma.metric, do_minimize=ma.mode=="min", random_seed=ma.random_seed, points_to_evaluate=ma.points_to_evaluate),
    Methods.CQR: lambda ma: CQR(config_space=ma.config_space, metric=ma.metric, do_minimize=ma.mode=="min", random_seed=ma.random_seed, points_to_evaluate=ma.points_to_evaluate),
    Methods.TPE: lambda ma: TPE(config_space=ma.config_space, metric=ma.metric, do_minimize=ma.mode=="min", random_seed=ma.random_seed, points_to_evaluate=ma.points_to_evaluate),
    Methods.REA: lambda ma: REA(config_space=ma.config_space, metric=ma.metric, do_minimize=ma.mode=="min", random_seed=ma.random_seed, points_to_evaluate=ma.points_to_evaluate),
    #Methods.OPT_CQR: lambda ma: OptformerScheduler(config_space=ma.config_space, metric=ma.metric, checkpoint_dir=Path(ma.checkpoint_dir), task_info={'name': ma.benchmark_name, 'algorithm': "CQR", 'metric_names': "feval"}, do_minimize=ma.mode=="min", random_seed=ma.random_seed, points_to_evaluate=ma.points_to_evaluate, n_sample_configurations=1),
    #Methods.OPT_REA: lambda ma: OptformerScheduler(config_space=ma.config_space, metric=ma.metric, checkpoint_dir=Path(ma.checkpoint_dir), task_info={'name': ma.benchmark_name, 'algorithm': "REA", 'metric_names': "feval"}, do_minimize=ma.mode=="min", random_seed=ma.random_seed, points_to_evaluate=ma.points_to_evaluate, n_sample_configurations=1),
    #Methods.OPT_BORE: lambda ma: OptformerScheduler(config_space=ma.config_space, metric=ma.metric, checkpoint_dir=Path(ma.checkpoint_dir), task_info={'name': ma.benchmark_name, 'algorithm': "BORE", 'metric_names': "feval"}, do_minimize=ma.mode=="min", random_seed=ma.random_seed, points_to_evaluate=ma.points_to_evaluate, n_sample_configurations=1),
    #Methods.OPT_TPE: lambda ma: OptformerScheduler(config_space=ma.config_space, metric=ma.metric, checkpoint_dir=Path(ma.checkpoint_dir), task_info={'name': ma.benchmark_name, 'algorithm': "TPE", 'metric_names': "feval"}, do_minimize=ma.mode=="min", random_seed=ma.random_seed, points_to_evaluate=ma.points_to_evaluate, n_sample_configurations=1),
    #Methods.OPT_CQR_TS: lambda ma: OptformerScheduler(config_space=ma.config_space, metric=ma.metric, checkpoint_dir=Path(ma.checkpoint_dir), task_info={'name': ma.benchmark_name, 'algorithm': "CQR", 'metric_names': "feval"}, do_minimize=ma.mode=="min", random_seed=ma.random_seed, points_to_evaluate=ma.points_to_evaluate, n_sample_configurations=50),
    #Methods.OPT_CQR_TS_5: lambda ma: OptformerScheduler(config_space=ma.config_space, metric=ma.metric, checkpoint_dir=Path(ma.checkpoint_dir), task_info={'name': ma.benchmark_name, 'algorithm': "CQR", 'metric_names': "feval"}, do_minimize=ma.mode=="min", random_seed=ma.random_seed, points_to_evaluate=ma.points_to_evaluate, n_sample_configurations=5),
}
