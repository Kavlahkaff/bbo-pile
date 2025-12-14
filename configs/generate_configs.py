
import yaml
from pathlib import Path

BASE_PATH_CLUSTER = '/data/horse/ws/aakl689g-optformer/syne-tune-benchmarks'
DATASET_NAME = 'concat'
WANDB_PROJECT = 'open_optformer_pythia_hp_sweep'

def generate_configs():
    model_names = ['pythia-5M', 'pythia-22M', 'pythia-72M']
    for model_name in model_names:
        base_config_path = Path(__file__).parent / f"{model_name}_template.yaml"

        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)

        token_counts = {
            "50M": 50_000_000,
            "100M": 100_000_000,
            "200M": 200_000_000,
            "400M": 400_000_000,
            "600M": 600_000_000,
            "800M": 800_000_000,
            "1B": 1_000_000_000,
        }
        lr_grid = {
            "1e-5": 1e-5,
            "3e-5": 3e-5,
            "6e-5": 6e-5,
            "1e-4": 1e-4,
            "3e-4": 3e-4,
            "6e-4": 6e-4,
            "1e-3": 1e-3,
        }

        base_path = Path(BASE_PATH_CLUSTER)

        for name, tokens in token_counts.items():
            for lr in lr_grid:
                new_config = base_config.copy()
                new_config['train']['max_tokens'] = tokens
                new_config['optimizer']['init_args']['lr'] = lr
                run_name = f"{model_name}_token_{name}_lr_{lr}"
                new_config['log']['run'] = run_name

                new_config['data']['init_args']['data_path'] = str(base_path / 'data' / 'tokenized_data' / DATASET_NAME)
                new_config['tokenizer_dir'] = str(base_path / 'tokenizer')
                new_config['out_dir'] = str(base_path / run_name)

                new_config['log']['project'] = WANDB_PROJECT

                new_filename = f"{run_name}.yaml"
                new_filepath = base_config_path.parent / new_filename

                with open(new_filepath, 'w') as f:
                    yaml.dump(new_config, f, sort_keys=False)

                print(f"Generated {new_filepath}")

if __name__ == "__main__":
    generate_configs()
