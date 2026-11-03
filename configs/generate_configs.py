"""
Generate litgpt training configs (grid over model size x token budget x learning rate x batch size x seed).

Without arguments this reproduces the original HPO sweep (v0.8, dataset 'all', vocab 1069). The GP experiments use
e.g. the token-budget / LR sweep of the 5M model on the reference export R:

    BASE_PATH=/projects/p_neurasearch/bbo-pile_experiments python configs/generate_configs.py \
        --version gp_v1 --dataset R --models qwen3_5M --tokens 300M 600M 1.2B 2.4B --lrs 5e-3 1e-2 --bsz 16 \
        --vocab_size 1062 --max_seq_length 1024 --seeds 0 --out_dir configs/gp
"""
import argparse
import copy
import math
import os
from pathlib import Path

import yaml

MODEL_SIZES = {
    'qwen3_2M': 2e6,
    'qwen3_5M': 5e6,
    'qwen3_10M': 10e6,
    'qwen3_13M': 13e6,
    'qwen3_30M': 30e6,
    'qwen3_80M': 80e6,
    'qwen3_150M': 150e6,
    'qwen3_450M': 450e6,
}
TOKEN_COUNTS = {
    "100M": 100_000_000, "200M": 200_000_000, "300M": 300_000_000, "400M": 400_000_000, "600M": 600_000_000,
    "800M": 800_000_000, "1B": 1_000_000_000, "1.2B": 1_200_000_000, "2B": 2_000_000_000,
    "2.4B": 2_400_000_000, "4B": 4_000_000_000,
}
LEGACY_VOCAB = {'v0.3': 1106, 'v0.4': 1072, 'v0.5': 1069, 'v0.6': 1069, 'v0.7': 1069, 'v0.8': 1069}


def parse_tokens(name: str) -> int:
    """'300M' -> 300_000_000, '1.2B' -> 1_200_000_000 (any number with suffix K, M or B)."""
    if name in TOKEN_COUNTS:
        return TOKEN_COUNTS[name]
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}[name[-1].upper()]
    return int(float(name[:-1]) * mult)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_path", default=os.environ.get("BASE_PATH"), help="default: $BASE_PATH")
    ap.add_argument("--version", default="v0.8", help="dataset / tokenizer version (subfolder)")
    ap.add_argument("--tokenizer_version", default=None, help="default: --version")
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--models", nargs="+", default=['qwen3_2M', 'qwen3_5M', 'qwen3_13M', 'qwen3_30M', 'qwen3_80M',
                                                    'qwen3_150M'])
    ap.add_argument("--tokens", nargs="+", default=["100M", "200M", "400M", "600M", "800M", "1B", "2B", "4B"])
    ap.add_argument("--lrs", nargs="+", default=["5e-3", "1e-2", "2e-2"])
    ap.add_argument("--bsz", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--vocab_size", type=int, default=None, help="default: the legacy value of --version")
    ap.add_argument("--max_seq_length", type=int, default=None, help="default: from the base config")
    ap.add_argument("--warmup", type=float, default=0.1, help="share of steps used for LR warm-up")
    ap.add_argument("--min_tokens_per_param", type=float, default=2.0, help="skip budgets below this ratio")
    ap.add_argument("--initial_checkpoint_dir", default=None, help="for fine-tuning runs (e.g. R -> O)")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--run_prefix", default="")
    ap.add_argument("--out_dir", default=str(Path(__file__).parent))
    return ap.parse_args()


def generate_configs(args):
    if args.base_path is None:
        raise SystemExit("set BASE_PATH or --base_path")
    base_path = Path(args.base_path)
    tok_version = args.tokenizer_version or args.version
    vocab = args.vocab_size or LEGACY_VOCAB.get(args.version)
    project = args.wandb_project or f'open_optformer_qwen3_hp_sweep_{args.version}'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counter = 0
    for model_name in args.models:
        model_size = MODEL_SIZES[model_name]
        with open(Path(__file__).parent / f"{model_name}.yaml", 'r') as f:
            base_config = yaml.safe_load(f)
        for name in args.tokens:
            tokens = parse_tokens(name)
            if tokens < args.min_tokens_per_param * model_size:
                # Do not run any model with smaller token budget of 2 times the parameter count
                continue
            for lr_name in args.lrs:
                lr = float(lr_name)
                for bsz in args.bsz:
                    for seed in args.seeds:
                        new_config = copy.deepcopy(base_config)
                        if args.max_seq_length:
                            new_config['train']['max_seq_length'] = args.max_seq_length
                            new_config['model_config']['block_size'] = args.max_seq_length
                        number_of_steps = tokens // (bsz * new_config['train']['max_seq_length'])
                        ws = int(number_of_steps * args.warmup)
                        print(model_name, name, number_of_steps, bsz, ws, str(ws / number_of_steps * 100) + '%')
                        new_config['optimizer']['init_args']['lr'] = lr
                        new_config['train']['max_tokens'] = tokens
                        new_config['train']['global_batch_size'] = bsz
                        new_config['train']['log_interval'] = math.ceil(number_of_steps / 200)
                        new_config['train']['lr_warmup_steps'] = ws
                        new_config['train']['micro_batch_size'] = bsz
                        new_config['train']['save_interval'] = math.ceil(number_of_steps / 10)  # 10 checkpoints
                        new_config['eval']['interval'] = math.ceil(number_of_steps / 50)

                        run_name = (f"{args.run_prefix}{model_name}_token_{name}_lr_{lr_name}_bsz_{bsz}_ws_{ws}"
                                    f"_seed_{seed}")
                        new_config['log']['run'] = run_name
                        new_config['log']['project'] = project
                        new_config['log']['group'] = model_name
                        new_config['seed'] = seed
                        new_config['data']['init_args']['data_path'] = str(
                            base_path / 'tokenized_dataset' / args.version / args.dataset)
                        new_config['tokenizer_dir'] = str(base_path / 'tokenizer' / tok_version)
                        new_config['out_dir'] = str(base_path / 'checkpoints' / args.version / run_name)
                        if args.initial_checkpoint_dir:
                            new_config['initial_checkpoint_dir'] = args.initial_checkpoint_dir
                        if vocab is not None:
                            new_config['model_config']['vocab_size'] = vocab
                            new_config['model_config']['padded_vocab_size'] = vocab
                        with open(out_dir / f"{run_name}.yaml", 'w') as f:
                            yaml.dump(new_config, f, sort_keys=False)
                        counter += 1

    print('num configs: ', counter)


if __name__ == "__main__":
    generate_configs(parse_args())
