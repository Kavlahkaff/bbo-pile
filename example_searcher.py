import time

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from syne_tune.util import catchtime
from syne_tune.backend.trial_status import Trial

from syne_tune.blackbox_repository.blackbox_surrogate import add_surrogate
from syne_tune.blackbox_repository import load_blackbox

from open_optformer.optformer_searcher import OptformerScheduler


#bb = load_blackbox("fcnet")["imagenet_resnet_batch_size_512"]
bb = load_blackbox("lcbench")["Fashion-MNIST"]
bb = add_surrogate(bb, predict_curves=False)
config_space = bb.configuration_space
objective = bb.objectives_names[0]

points_to_evaluate = [
    {
        k: v.sample(random_state=np.random.RandomState(0)) if hasattr(v, "sample") else v
        for k, v in config_space.items()
    }
    for _ in range(1)
]

print(points_to_evaluate[0])

name = "remove-forward-refactor"

checkpoint_dir = pathlib.Path("./checkpoint/")
searcher = OptformerScheduler(
    config_space=config_space,
    checkpoint_dir=checkpoint_dir,
    metric=objective,
    random_seed=0,
    task_info={'name': 'pd1_imagenet_resnet_batch_size_512',
            'algorithm': "CQR",
            'metric_names': objective},
    points_to_evaluate=points_to_evaluate
)


# Store runtimes for each trial
runtimes = {}
n = 20
for trial_id in range(n):
    # Start timer for this trial
    start_time = time.time()

    print(f"Trial: {trial_id}")
    trial_suggestion = searcher.suggest()
    config = trial_suggestion.config
    print("Config: ", config)
    metric = bb(config, fidelity=10)[objective]
    print("Metric: ", metric)
    searcher.on_trial_complete(Trial(trial_id=trial_id, config=config, creation_time=time.time()), {objective: metric})

    # Calculate runtime for this trial
    runtime = time.time() - start_time
    runtimes[trial_id] = runtime
    print(f"Runtime: {runtime:.4f} seconds\n")

# Plot the runtimes
plt.figure(figsize=(10, 6))
plt.plot(list(runtimes.keys()), list(runtimes.values()), marker='o', linewidth=2, markersize=8)
plt.xlabel('Trial ID', fontsize=12)
plt.ylabel('Runtime (seconds)', fontsize=12)
plt.title('Runtime per Trial', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add value labels on each point
for i, runtime in runtimes.items():
    plt.text(i, runtime, f'{runtime:.3f}s', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f"fig-{name}.png")
pd.Series(runtimes).to_csv(f"data-{name}.csv", index=False)

# Print summary statistics
print("\n=== Runtime Summary ===")
print(f"Total runtime: {sum(runtimes):.4f} seconds")
print(f"Average runtime: {sum(runtimes) / len(runtimes):.4f} seconds")
print(f"Min runtime: {min(runtimes):.4f} seconds")
print(f"Max runtime: {max(runtimes):.4f} seconds")