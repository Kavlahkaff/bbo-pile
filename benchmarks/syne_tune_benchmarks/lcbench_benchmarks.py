from benchmark_definitions import BenchmarkDefinition, n_full_evals
from syne_tune.blackbox_repository import load_blackbox


def lcbench_benchmark(dataset_name):
    return BenchmarkDefinition(
        max_wallclock_time=36000,
        max_num_evaluations=n_full_evals,
        n_workers=4,
        elapsed_time_attr="time",
        metric="val_accuracy",
        mode="max",
        blackbox_name="lcbench",
        dataset_name=dataset_name,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
    )


lcbench_benchmark_definitions = {}

blackboxes = load_blackbox('lcbench')
for ds in list(blackboxes.keys()):
    lcbench_benchmark_definitions["lcbench_" + ds] = lcbench_benchmark(ds)
