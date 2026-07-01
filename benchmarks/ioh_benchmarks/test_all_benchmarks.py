import os
import sys

sys.path.append(os.path.abspath("../../"))

from benchmarks_definitions import benchmark_definitions
from baselines import methods, MethodArguments
from syne_tune.optimizer.schedulers.ask_tell_scheduler import AskTellScheduler

def test_all():
    success_count = 0
    total = len(benchmark_definitions)
    print(f"Testing {total} benchmarks...")
    
    for name, blackbox in benchmark_definitions.items():
        cs = blackbox.configuration_space
        try:
            scheduler = methods["RS"](MethodArguments(
                metric="y", random_seed=42, mode="min", config_space=cs,
                points_to_evaluate=[{k: v.sample() for k, v in cs.items()}],
                checkpoint_dir="", benchmark_name=name
            ))
            
            ask_tell = AskTellScheduler(base_scheduler=scheduler)
            trial = ask_tell.ask()
            ask_tell.tell(trial, blackbox(trial.config))
            success_count += 1
                
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    print(f"Test Summary: {success_count}/{total} successful.")

if __name__ == "__main__":
    test_all()
