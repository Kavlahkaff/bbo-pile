import time

import pathlib

from syne_tune.backend.trial_status import Trial

from syne_tune.blackbox_repository.blackbox_surrogate import add_surrogate
from syne_tune.blackbox_repository import load_blackbox

from open_optformer.optformer_searcher import OptformerScheduler


bb = load_blackbox("pd1")["imagenet_resnet_batch_size_512"]
bb = add_surrogate(bb, predict_curves=False)
config_space = bb.configuration_space
objective = bb.objectives_names[0]

points_to_evaluate = [
    {
        k: v.sample() if hasattr(v, "sample") else v
        for k, v in config_space.items()
    }
    for _ in range(1)
]

checkpoint_dir = pathlib.Path("./checkpoint/")
searcher = OptformerScheduler(config_space=config_space, checkpoint_dir=checkpoint_dir,
                              metric=objective,
                             task_info={'name': 'pd1_imagenet_resnet_batch_size_512',
                                        'algorithm': "CQR",
                                        'metric_names': objective},
                             points_to_evaluate=points_to_evaluate)

for trial_id in range(50):
    print(f"Trial: {trial_id}")
    trial_suggestion = searcher.suggest()
    config = trial_suggestion.config
    print("Config: ", config)
    metric = bb(config, fidelity={'global_step': 251})[objective]
    print("Metric: ", metric)
    searcher.on_trial_complete(Trial(trial_id=trial_id, config=config, creation_time=time.time()), {objective: metric})

