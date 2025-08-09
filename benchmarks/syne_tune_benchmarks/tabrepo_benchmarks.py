from blackbox_benchmarks import BenchmarkDefinition
from syne_tune.blackbox_repository import load_blackbox

n_full_evals = 100

def tabrepo_benchmark(blackbox_name: str, dataset_name: str):
    return BenchmarkDefinition(
        max_wallclock_time=36000,
        max_num_evaluations=1 * n_full_evals,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",  # todo should also include time_train_s + time_infer_s as metric
        metric="metric_error_val",  # could also do rank
        mode="min",
        blackbox_name=blackbox_name,
        dataset_name=dataset_name,
        use_surrogate=True,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
    )

tabrepo_search_spaces = [
    'tabrepo_RandomForest',
    'tabrepo_LinearModel',
    'tabrepo_CatBoost',
    'tabrepo_XGBoost',
    'tabrepo_ExtraTrees',
    'tabrepo_NeuralNetTorch',
    'tabrepo_LightGBM',
    'tabrepo_KNeighbors'
]

tabrepo_benchmark_definitions = {}

#Currently runs only a subset of tabrepo. Adapt as needed.
for ss in tabrepo_search_spaces[:1]:
    blackboxes = load_blackbox(ss)
    for ds in list(blackboxes.keys())[:11]:
        tabrepo_benchmark_definitions[ss + "_" + ds] = tabrepo_benchmark(ss, ds)
