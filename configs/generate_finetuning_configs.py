import os
import math
import yaml
from pathlib import Path

BASE_PATH_CLUSTER = os.environ['BASE_PATH']
VERSION = 'v0.9'
SEED = 0
WANDB_PROJECT = f'open_optformer_qwen3_finetune_sweep_{VERSION}'

# Fraction of the pretraining token budget to use for fine-tuning.
# Pulled from the winning pretrain run's own `train.max_tokens`, so this
# is relative rather than a fixed absolute grid across model sizes.
TOKEN_FRACTIONS = {"1_5th": 1 / 5, "1_6th": 1 / 6}

LR_GRID = {"5e-5": 5e-5, "2e-4": 2e-4, "1e-3": 1e-3}
WARMUP_FRACTIONS = {"0pct": 0.0, "10pct": 0.10}
BSZ_GRID = [4, 8, 16]

# Minimum number of optimizer steps a config must produce to be worth
# running at all (guards against tiny token-budget / large-bsz combos
# degenerating to a handful of steps).
MIN_STEPS = 50


def generate_configs():

    # Map each model size to the checkpoint directory of its best
    # (lowest val-loss) pretraining run. Each of these directories is
    # expected to contain final/{lit_model.pth, hyperparameters.yaml,
    # model_config.yaml, tokenizer.model} as produced by retrain.py's
    # `setup()` / save_checkpoint().
    best_run_names = {
        'qwen3_2M': 'qwen3_2M_token_2B_lr_5e-3_bsz_4_seed_0',
        'qwen3_5M': 'qwen3_5M_token_2B_lr_5e-3_bsz_16_seed_0',
        'qwen3_13M': 'qwen3_13M_token_2B_lr_5e-3_bsz_16_seed_0',
        'qwen3_30M': 'qwen3_30M_token_2B_lr_5e-3_bsz_4_seed_0',
        'qwen3_80M': 'qwen3_80M_token_2B_lr_5e-3_bsz_16_seed_0',
    }
    best_checkpoints = {
        model_name: Path('/projects/p_neurasearch/bbo-pile_experiments') / 'checkpoints' / 'v0.8' / run_name / 'final'
        for model_name, run_name in best_run_names.items()
    }

    out_root = Path(__file__).parent
    counter = 0

    for model_name, checkpoint_dir in best_checkpoints.items():
        hparams_path = checkpoint_dir / 'hyperparameters.yaml'

        if not hparams_path.exists():
            print(f"[skip] {model_name}: no hyperparameters.yaml at {hparams_path} (fill in the real path)")
            continue

        with open(hparams_path, 'r') as f:
            base_config = yaml.safe_load(f)

        # Everything under model_config is tied to the trained weights
        # and must not be touched -- it's inherited verbatim from the
        # checkpoint's own hyperparameters.yaml.
        pretrain_tokens = base_config['train']['max_tokens']
        max_seq_length = base_config['train']['max_seq_length']

        base_path = Path(BASE_PATH_CLUSTER)

        for frac_name, frac in TOKEN_FRACTIONS.items():
            tokens = int(pretrain_tokens * frac)

            for lr_name, lr in LR_GRID.items():
                for bsz in BSZ_GRID:
                    for wu_name, wu_frac in WARMUP_FRACTIONS.items():

                        number_of_steps = tokens // (bsz * max_seq_length)
                        if number_of_steps < MIN_STEPS:
                            continue

                        new_config = base_config.copy()

                        ws = int(number_of_steps * wu_frac)

                        new_config['optimizer']['init_args']['lr'] = lr

                        new_config['train']['max_tokens'] = tokens
                        new_config['train']['global_batch_size'] = bsz
                        new_config['train']['micro_batch_size'] = bsz
                        new_config['train']['log_interval'] = math.ceil(number_of_steps / 200)
                        new_config['train']['lr_warmup_steps'] = ws
                        new_config['train']['save_interval'] = math.ceil(number_of_steps / 10)
                        new_config['eval']['interval'] = math.ceil(number_of_steps / 50)

                        # Load pretrained weights only; optimizer state,
                        # iter_num and step_count all start fresh. Mutually
                        # exclusive with `resume`, which must stay False.
                        new_config['initial_checkpoint_dir'] = str(checkpoint_dir)
                        new_config['resume'] = False

                        run_name = (
                            f"{model_name}_ft_tok_{frac_name}_lr_{lr_name}"
                            f"_bsz_{bsz}_wu_{wu_name}_seed_{SEED}"
                        )

                        new_config['log']['run'] = run_name
                        new_config['log']['project'] = WANDB_PROJECT
                        new_config['log']['group'] = f"{model_name}_finetune"
                        new_config['seed'] = SEED

                        # Same dataset, same train/val split, same tokenizer
                        # as pretraining -- left untouched from base_config.
                        # (Only the mock label used inside that data path
                        # differs, which requires no config changes here.)

                        new_config['out_dir'] = str(
                            base_path / 'checkpoints_finetune' / VERSION / run_name
                        )

                        new_filename = f"{run_name}.yaml"
                        new_filepath = out_root / new_filename

                        with open(new_filepath, 'w') as f:
                            yaml.dump(new_config, f, sort_keys=False)

                        counter += 1

    print('num configs: ', counter)


if __name__ == "__main__":
    generate_configs()
