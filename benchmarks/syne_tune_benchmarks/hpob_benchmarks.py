from benchmark_definitions import BenchmarkDefinition, n_full_evals


def hpob_benchmark(blackbox_name: str, dataset_name: str):
    return BenchmarkDefinition(
        max_wallclock_time=36000,
        max_num_evaluations=1 * n_full_evals,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_accuracy",
        mode="max",
        blackbox_name=blackbox_name,
        dataset_name=dataset_name,
        use_surrogate=True,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
    )

hpob_search_spaces = [
    "hpob_4796",
    "hpob_5527",
    "hpob_5636",
    "hpob_5859",
    "hpob_5860",
    "hpob_5891",
    "hpob_5906",
    "hpob_5965",
    "hpob_5970",
    "hpob_5971",
    "hpob_6766",
    "hpob_6767",
    "hpob_6794",
    "hpob_7607",
    "hpob_7609",
    "hpob_5889",
]

hpob_benchmark_definitions = {}

for ss in hpob_search_spaces:
    from syne_tune.blackbox_repository import load_blackbox

    blackboxes = load_blackbox(ss, local_files_only=True)
    for ds in list(blackboxes.keys()):
        hpob_benchmark_definitions[ss + "_" + ds] = hpob_benchmark(ss, ds)
