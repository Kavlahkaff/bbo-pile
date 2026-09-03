"""Generate pretrain configs with a CONSTANT learning rate (no cosine
annealing during pretraining), at a fixed 2B token budget.

litgpt's LR schedule (open_optformer/training/pretrain.py's `get_lr`) is
`min_lr + coeff * (lr - min_lr)`, cosine-decayed from `lr` down to
`train.min_lr` after warmup. Setting `train.min_lr == optimizer.lr` makes
`coeff` irrelevant -- the LR stays flat at `lr` for the rest of pretraining
(warmup still ramps up to `lr` as normal). The intended annealing then
happens later, during finetuning: finetune on this checkpoint as usual (see
experiments/only_best/finetune.sh or experiments/advantage_sft/finetune.sh)
with its own (much lower) learning rate -- that finetuning stage acts as the
decay/"cooldown" phase, using the finetuning data mixture instead of a
cosine schedule baked into pretraining. This is the ablation debugged in
2026-09-03's meeting notes ("pretraining without lr-annealing").

Reuses each model size's already-established best (lr, batch_size) from the
main sweep (see checkpoint paths in sample_distribution_opt_best.sh /
configs/finetune/*.yaml) rather than re-sweeping -- this is a single
targeted ablation, not a new hyperparameter search.

Usage:
    export BASE_PATH=/data/horse/ws/luth474h-bbo-pile-experiments/masters-thesis/experiments
    cd configs && python generate_configs_constant_lr.py
"""
import os
import math
import yaml
from pathlib import Path

BASE_PATH_CLUSTER = os.environ['BASE_PATH']
DATASET_NAME = 'all'
VERSION = 'v0.9'
SEED = 0
WANDB_PROJECT = f'open_optformer_qwen3_hp_sweep_{VERSION}_constant_lr'
TOKENS = 2_000_000_000  # 2B, per the ablation's fixed token budget

# model_name -> (lr, global/micro batch size), matching each size's
# established-best pretrain run (see PRETRAINED_CKPTS in
# analysis/best_baseline_comparison/sample_distribution_opt_best.sh).
# 150M has no established-best combo yet (never swept) -- uses the same
# (lr, bsz) as the other sizes as a reasonable default, not a tuned choice.
MODEL_CONFIGS = {
    'qwen3_2M':  dict(size=2e6,   lr=5e-3, lr_name='5e-3', bsz=4),
    'qwen3_5M':  dict(size=5e6,   lr=5e-3, lr_name='5e-3', bsz=16),
    'qwen3_13M': dict(size=13e6,  lr=5e-3, lr_name='5e-3', bsz=16),
    'qwen3_30M': dict(size=30e6,  lr=5e-3, lr_name='5e-3', bsz=4),
    'qwen3_80M': dict(size=80e6,  lr=5e-3, lr_name='5e-3', bsz=16),
    'qwen3_150M': dict(size=150e6, lr=5e-3, lr_name='5e-3', bsz=16),
}


def generate_configs():
    base_path = Path(BASE_PATH_CLUSTER)
    out_dir = Path(__file__).parent / "pretrain_constant_lr"
    out_dir.mkdir(exist_ok=True)

    counter = 0
    for model_name, cfg in MODEL_CONFIGS.items():
        base_config_path = Path(__file__).parent / f"{model_name}.yaml"
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)

        lr, lr_name, bsz = cfg['lr'], cfg['lr_name'], cfg['bsz']
        number_of_steps = TOKENS // (bsz * base_config['train']['max_seq_length'])
        ws = int(number_of_steps * 0.1)  # 10% warm-up, same convention as generate_configs.py

        new_config = base_config.copy()
        new_config['optimizer']['init_args']['lr'] = lr
        new_config['train']['max_tokens'] = TOKENS
        new_config['train']['global_batch_size'] = bsz
        new_config['train']['micro_batch_size'] = bsz
        new_config['train']['log_interval'] = math.ceil(number_of_steps / 200)
        new_config['train']['lr_warmup_steps'] = ws
        new_config['train']['save_interval'] = math.ceil(number_of_steps / 10)
        new_config['train']['min_lr'] = lr  # <-- the whole point: no annealing
        new_config['eval']['interval'] = math.ceil(number_of_steps / 50)

        run_name = f"{model_name}_token_2B_lr_{lr_name}_bsz_{bsz}_constlr_seed_{SEED}"
        new_config['log']['run'] = run_name
        new_config['log']['project'] = WANDB_PROJECT
        new_config['log']['group'] = model_name
        new_config['seed'] = SEED
        new_config['data']['init_args']['data_path'] = str(base_path / 'tokenized_dataset' / VERSION / DATASET_NAME)
        new_config['tokenizer_dir'] = str(base_path / 'tokenizer' / VERSION)
        new_config['out_dir'] = str(base_path / 'checkpoints' / VERSION / 'constant_lr' / run_name)
        new_config['model_config']['vocab_size'] = 1061
        new_config['model_config']['padded_vocab_size'] = 1061

        new_filepath = out_dir / f"{run_name}.yaml"
        with open(new_filepath, 'w') as f:
            yaml.dump(new_config, f, sort_keys=False)

        print(f"{model_name}: lr={lr} min_lr={lr} bsz={bsz} steps={number_of_steps} ws={ws} -> {new_filepath}")
        counter += 1

    print('num configs: ', counter)


if __name__ == "__main__":
    generate_configs()
