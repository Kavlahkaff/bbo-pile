import os
import math
import yaml
from pathlib import Path

BASE_PATH_CLUSTER = os.environ['BASE_PATH']
DATASET_NAME = 'all'
VERSION = 'v0.4'
SEED = 0
WANDB_PROJECT = f'open_optformer_qwen3_hp_sweep_{VERSION}'

def generate_configs():
    model_names = ['qwen3_5M', 'qwen3_10M', 'qwen3_20M',
                   'qwen3_30M', 'qwen3_50M', 'qwen3_100M',
                   'qwen3_200M', 'qwen3_500M']
    for model_name in model_names:
        base_config_path = Path(__file__).parent / f"{model_name}.yaml"

        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)

        token_counts = {
            "100M": 100_000_000,
            "200M": 200_000_000,
            "400M": 400_000_000,
            "600M": 600_000_000,
            "800M": 800_000_000,
            "1B": 1_000_000_000,
            '2B': 2_000_000_000,
            '4B': 4_000_000_000,
        }
        lr_grid = {"5e-4": 5e-4, "1e-3": 1e-3, "5e-3": 5e-3, "1e-2": 1e-2}

#        gas_grid = [1, 2, 4, 8, 16]
#        mbs = 16
        bsz_grid = [4, 8, 16, 32]
        base_path = Path(BASE_PATH_CLUSTER)
        for name, tokens in token_counts.items():
            for lr_name, lr in lr_grid.items():
                for bsz in bsz_grid:

#                    bsz = int(gas * mbs)
                    number_of_steps = tokens // (bsz * base_config['train']['max_seq_length'])
                    new_config = base_config.copy()

                    new_config['optimizer']['init_args']['lr'] = lr
                    
                    new_config['train']['max_tokens'] = tokens
                    new_config['train']['global_batch_size'] = bsz
                    new_config['train']['log_interval'] = math.ceil(number_of_steps / 50)
                    new_config['train']['lr_warmup_steps'] = int(number_of_steps * 0.05)  # 5% warmup
                    new_config['train']['micro_batch_size'] = bsz
                    new_config['train']['save_interval'] = math.ceil(number_of_steps / 10)  # Save 10 checkpoints per model
                    new_config['eval']['interval'] = math.ceil(number_of_steps / 20)

                    run_name = f"{model_name}_token_{name}_lr_{lr_name}_bsz_{bsz}_seed_{SEED}"
                    new_config['log']['run'] = run_name
                    new_config['log']['project'] = WANDB_PROJECT
                    new_config['log']['group'] = model_name
                    new_config['seed'] = SEED
                    new_config['data']['init_args']['data_path'] = str(base_path / 'tokenized_dataset' / VERSION / DATASET_NAME)
                    new_config['tokenizer_dir'] = str(base_path / 'tokenizer' / VERSION )
                    new_config['out_dir'] = str(base_path / 'checkpoints' / VERSION /  run_name)
                    if VERSION == 'v0.3':
                        new_config['model_config']['vocab_size'] = 1106
                        new_config['model_config']['padded_vocab_size'] = 1106
                    elif VERSION == 'v0.4':
                        new_config['model_config']['vocab_size'] = 1072
                        new_config['model_config']['padded_vocab_size'] = 1072
                    new_filename = f"{run_name}.yaml"
                    new_filepath = base_config_path.parent / new_filename

                    with open(new_filepath, 'w') as f:
                        yaml.dump(new_config, f, sort_keys=False)

                    print(f"Generated {new_filepath}")

if __name__ == "__main__":
    generate_configs()
