"""Pretrain config for the quantile (rank) y-normalization experiment
(experiments/quantile_norm/). Same schedule formulas as
generate_configs_true_range.py (10% warmup, 10 checkpoints, 50 evals, v0.8
tokenizer vocab 1069) but restricted to the single grid point the experiment
actually trains -- qwen3_80M, 2B tokens, lr 5e-3, bsz 16 -- so that
configs/pretrain_quantile/ holds exactly the file experiments/quantile_norm/
pretrain.sh consumes instead of the full ~130-file sweep. Widen the grids
below if more sizes are ever needed.

Usage:
    export BASE_PATH=/data/cat/ws/luth474h-bbo-pile-expert-iteration/masters-thesis/experiments
    cd configs && python generate_configs_quantile.py
"""
import os
import math
import yaml
from pathlib import Path

BASE_PATH_CLUSTER = os.environ['BASE_PATH']
DATASET_NAME = 'all-quantile'
VERSION = 'v0.8'
SEED = 0
WANDB_PROJECT = f'open_optformer_qwen3_hp_sweep_{VERSION}_quantile'
OUT_SUBDIR = 'quantile'


def generate_configs():
    model_names = {
        'qwen3_80M': 80e6,
    }
    token_counts = {
        '2B': 2_000_000_000,
    }
    lr_grid = {"5e-3": 5e-3}
    bsz_grid = [16]

    out_dir = Path(__file__).parent / "pretrain_quantile"
    out_dir.mkdir(exist_ok=True)
    base_path = Path(BASE_PATH_CLUSTER)

    counter = 0
    for model_name, model_size in model_names.items():
        base_config_path = Path(__file__).parent / f"{model_name}.yaml"
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)

        for name, tokens in token_counts.items():
            if tokens < 2 * model_size:
                continue

            for lr_name, lr in lr_grid.items():
                for bsz in bsz_grid:
                    number_of_steps = tokens // (bsz * base_config['train']['max_seq_length'])
                    new_config = base_config.copy()

                    ws = int(number_of_steps * 0.1)
                    print(model_name, name, number_of_steps, bsz, ws, str(ws / number_of_steps * 100) + '%')
                    new_config['optimizer']['init_args']['lr'] = lr

                    new_config['train']['max_tokens'] = tokens
                    new_config['train']['global_batch_size'] = bsz
                    new_config['train']['log_interval'] = math.ceil(number_of_steps / 200)
                    new_config['train']['lr_warmup_steps'] = ws
                    new_config['train']['micro_batch_size'] = bsz
                    new_config['train']['save_interval'] = math.ceil(number_of_steps / 10)
                    new_config['eval']['interval'] = math.ceil(number_of_steps / 50)

                    run_name = f"{model_name}_token_{name}_lr_{lr_name}_bsz_{bsz}_ws_{ws}_seed_{SEED}"

                    new_config['log']['run'] = run_name
                    new_config['log']['project'] = WANDB_PROJECT
                    new_config['log']['group'] = model_name
                    new_config['seed'] = SEED
                    new_config['data']['init_args']['data_path'] = str(base_path / 'data' / 'tokenized_data' / DATASET_NAME)
                    new_config['tokenizer_dir'] = str(base_path / 'tokenizer' / VERSION)
                    new_config['out_dir'] = str(base_path / 'pretrained' / OUT_SUBDIR / run_name)
                    new_config['model_config']['vocab_size'] = 1069
                    new_config['model_config']['padded_vocab_size'] = 1069

                    with open(out_dir / f"{run_name}.yaml", 'w') as f:
                        yaml.dump(new_config, f, sort_keys=False)
                    counter += 1

    print('num configs: ', counter)


if __name__ == "__main__":
    generate_configs()
