#!/usr/bin/env python3
"""
Take existing pretrain YAML configs and generate a FINETUNE GRID SWEEP from
them, mirroring the structure of the original pretrain sweep generator
(model_names x token_counts x lr_grid x bsz_grid), but restricted to the
parameters that actually make sense to re-sweep when finetuning:

  - learning rate  (finetuning is sensitive to this; pretrain-tuned lr is
                     usually too high once weights are no longer random init)
  - epochs         (stands in for the pretrain sweep's token-budget grid --
                     here the dataset is fixed size, so epochs controls
                     exposure instead)

NOT swept (kept exactly as in each model's base pretrain config, since these
were already tuned per model size during the pretrain sweep and re-sweeping
them here would just multiply config count without real benefit):
  - global_batch_size / micro_batch_size
  - architecture (model_config)

For each (config, lr, epochs) combination this script:
- points data/checkpoint/tokenizer/out_dir at the finetune dataset & pretrained checkpoint
- recomputes max_tokens, save_interval, eval.interval, log_interval, lr_warmup_steps
  from --train-tokens using that config's own global_batch_size/block_size, following
  the SAME formulas as the pretrain generator (fractions of total run steps):
      warmup_steps  = int(total_steps * warmup_fraction)   # default 10%, matches pretrain
      log_interval  = ceil(total_steps / num_logs)         # default 200 logs/run
      save_interval = ceil(total_steps / num_checkpoints)  # default 10 checkpoints/run
      eval.interval = ceil(total_steps / num_evals)        # default 50 evals/run
- sets lr to an ABSOLUTE value from --lr-grid (not a multiplier on the base config's
  existing lr -- multiplying an already-small base lr silently compounds into a
  vanishingly small value, e.g. 0.1 x 5e-5 = 5e-6). Default grid is deliberately
  well below the ~5e-3 pretrain lr, but not so small training barely moves:
  [5e-4, 1e-4, 5e-5].
- leaves model_config (architecture) completely untouched

Model checkpoint directory names are auto-derived from the config filename by
stripping the "_ws_<digits>" segment, matching the CONFIGS/MODELS convention
used in the sbatch array script, e.g.:
    qwen3_13M_token_2B_lr_5e-3_bsz_16_ws_3051_seed_0.yaml
      -> qwen3_13M_token_2B_lr_5e-3_bsz_16_seed_0

Usage:
    python apply_finetune_schedule.py \
        --configs finetune/*.yaml \
        --train-tokens 478669284 \
        --epochs-grid 1 2 3 \
        --lr-grid 5e-4 1e-4 5e-5 \
        --finetune-data-path /data/horse/ws/luth474h-master_thesis/experiments/data/tokenized_data/only-best \
        --checkpoint-base /projects/p_neurasearch/bbo-pile_experiments/checkpoints/v0.8 \
        --tokenizer-dir /projects/p_neurasearch/bbo-pile_experiments/tokenizer/v0.8 \
        --out-base /data/horse/ws/luth474h-master_thesis/experiments/finetuned/only-best \
        --output-dir finetuning_only_best \
        --project open_optformer_qwen3_finetune_only_best_v0.9

For a single non-swept config, just pass one value each: --epochs-grid 1 --lr-grid 1e-4

Add --dry-run to preview changes without writing files.
"""

import argparse
import glob
import math
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("error: PyYAML not found. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def derive_model_name(config_path: Path) -> str:
    return re.sub(r"_ws_\d+", "", config_path.stem)


def derive_group(model_name: str) -> str:
    m = re.match(r"^(qwen3_\d+M)", model_name)
    return m.group(1) if m else model_name


def expand_configs(patterns):
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(m) for m in sorted(matches))
        elif Path(pattern).exists():
            paths.append(Path(pattern))
        else:
            print(f"[warn] no match for {pattern}", file=sys.stderr)
    return paths


def lr_label(lr: float) -> str:
    """Format like the pretrain generator's grid keys, e.g. 0.0005 -> '5e-4'."""
    s = f"{lr:.0e}"          # e.g. '5e-04'
    mantissa, exp = s.split("e")
    exp = int(exp)
    return f"{mantissa}e{exp}"  # '5e-4'


def epoch_label(epochs: float) -> str:
    return f"{epochs:g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True, help="pretrain config yaml paths or glob patterns")
    ap.add_argument("--train-tokens", type=int, required=True, help="total tokens in finetune train split")
    ap.add_argument("--epochs-grid", type=float, nargs="+", default=[1, 2, 3],
                     help="epoch counts to sweep over (default: 1 2 3)")
    ap.add_argument("--lr-grid", type=float, nargs="+", default=[5e-4, 1e-4, 5e-5],
                     help="ABSOLUTE learning rates to sweep over -- not a multiplier on the base "
                          "config's lr (default: 5e-4 1e-4 5e-5). Keep these comfortably below the "
                          "pretrain lr (~5e-3) but not vanishingly small.")
    ap.add_argument("--finetune-data-path", required=True, help="new data.init_args.data_path")
    ap.add_argument("--checkpoint-base", required=True,
                     help="dir containing <model_name>/final pretrained checkpoints")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--out-base", required=True, help="new out_dir will be <out-base>/<run_name>")
    ap.add_argument("--output-dir", required=True, help="where to write the new finetune yaml files")
    ap.add_argument("--project", default=None, help="override log.project for all configs")
    ap.add_argument("--seed", type=int, default=None, help="override seed for all configs (default: keep base config's)")
    # Defaults match the original sweep generator's convention (fractions of TOTAL run steps):
    ap.add_argument("--num-checkpoints", type=int, default=10, help="save_interval = ceil(total_steps / this)")
    ap.add_argument("--num-evals", type=int, default=50, help="eval.interval = ceil(total_steps / this)")
    ap.add_argument("--num-logs", type=int, default=200, help="log_interval = ceil(total_steps / this)")
    ap.add_argument("--warmup-fraction", type=float, default=0.1, help="lr_warmup_steps = int(total_steps * this)")
    ap.add_argument("--dry-run", action="store_true", help="print planned changes without writing files")
    args = ap.parse_args()

    for lr in args.lr_grid:
        if lr >= 5e-3:
            print(f"[warn] --lr-grid value {lr:.1e} is >= typical pretrain lr (5e-3) -- "
                  f"probably too high for finetuning a pretrained checkpoint.", file=sys.stderr)
        if lr < 1e-6:
            print(f"[warn] --lr-grid value {lr:.1e} is very small -- training may barely move.", file=sys.stderr)

    config_paths = expand_configs(args.configs)
    if not config_paths:
        print("error: no config files found", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for path in config_paths:
        with open(path) as f:
            base_cfg = yaml.safe_load(f)

        try:
            gbs = base_cfg["train"]["global_batch_size"]
        except KeyError:
            print(f"[warn] {path}: missing train.global_batch_size, skipping", file=sys.stderr)
            continue
        block_size = (base_cfg.get("train", {}).get("max_seq_length")
                      or base_cfg.get("model_config", {}).get("block_size"))
        if block_size is None:
            print(f"[warn] {path}: missing block_size, skipping", file=sys.stderr)
            continue

        model_name = derive_model_name(path)
        group_name = derive_group(model_name)
        orig_min_lr = base_cfg["train"].get("min_lr", None)
        orig_lr = base_cfg["optimizer"]["init_args"]["lr"]
        try:
            min_lr_ratio = float(orig_min_lr) / float(orig_lr) if orig_min_lr is not None else 0.2
        except (TypeError, ValueError, ZeroDivisionError):
            min_lr_ratio = 0.2

        tokens_per_step = gbs * block_size
        for epochs in args.epochs_grid:
            for lr in args.lr_grid:
                cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # cheap deep copy

                steps_per_epoch = args.train_tokens / tokens_per_step
                total_steps = int(steps_per_epoch * epochs)
                max_tokens = int(round(total_steps * tokens_per_step))
                save_interval = max(1, math.ceil(total_steps / args.num_checkpoints))
                eval_interval = max(1, math.ceil(total_steps / args.num_evals))
                log_interval = max(1, math.ceil(total_steps / args.num_logs))
                warmup_steps = max(1, int(total_steps * args.warmup_fraction))
                min_lr = lr * min_lr_ratio

                run_name = f"{model_name}_ft_epochs_{epoch_label(epochs)}_lr_{lr_label(lr)}"
                seed = args.seed if args.seed is not None else cfg.get("seed", 0)
                run_name += f"_seed_{seed}"

                # Force vocabulary size to match the v0.8 pretrained checkpoint
                if "model_config" in cfg and "padded_vocab_size" in cfg["model_config"]:
                    cfg["model_config"]["padded_vocab_size"] = 1069

                cfg["data"]["init_args"]["data_path"] = args.finetune_data_path
                cfg["data"]["init_args"]["split_names"] = ["train", "valid"]

                cfg["initial_checkpoint_dir"] = f"{args.checkpoint_base}/{model_name}/final"
                cfg["tokenizer_dir"] = args.tokenizer_dir
                cfg["out_dir"] = f"{args.out_base}/{run_name}"
                cfg["resume"] = False
                cfg["seed"] = seed

                cfg["optimizer"]["init_args"]["lr"] = lr

                cfg["train"]["min_lr"] = min_lr
                cfg["train"]["max_tokens"] = max_tokens
                cfg["train"]["max_steps"] = None
                cfg["train"]["save_interval"] = save_interval
                cfg["train"]["log_interval"] = log_interval
                cfg["train"]["lr_warmup_steps"] = warmup_steps
                # global_batch_size / micro_batch_size / block_size / model_config: untouched

                cfg["eval"]["interval"] = eval_interval

                if args.project:
                    cfg["log"]["project"] = args.project
                cfg["log"]["run"] = run_name
                cfg["log"]["group"] = group_name

                summary = (f"{run_name}: gbs={gbs} block={block_size} steps/epoch={steps_per_epoch:,.1f} "
                           f"total_steps={total_steps:,} max_tokens={max_tokens:,} save_interval={save_interval:,} "
                           f"eval.interval={eval_interval:,} log_interval={log_interval:,} "
                           f"warmup={warmup_steps:,} lr={lr:.1e} min_lr={min_lr:.1e}")
                print(summary)

                if not args.dry_run:
                    out_path = out_dir / f"{run_name}.yaml"
                    with open(out_path, "w") as f:
                        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
                n_written += 1

    print(f"\n{'[dry-run] would write' if args.dry_run else 'wrote'} {n_written} config(s) "
          f"({len(config_paths)} model(s) x {len(args.epochs_grid)} epoch value(s) x {len(args.lr_grid)} lr value(s))"
          + (f" to {out_dir}/" if not args.dry_run else ""))
    if args.dry_run:
        print("Remove --dry-run to write configs.")


if __name__ == "__main__":
    main()
