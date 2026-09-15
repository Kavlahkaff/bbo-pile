"""Counts the total number of tokens in a litdata-chunked, TokensLoader-format
dataset directory -- i.e. anything produced by preprocess_data.py (see its
`ld.optimize(..., item_loader=ld.TokensLoader(...))` call). Works on the
only-best dataset, the full "all" dataset, or any other variant under
data/tokenized_data/ (causal, true-range, constant-lr, ...) -- anywhere a
TRAIN_TOKENS value is needed for apply_finetune_schedule.py's --train-tokens
(see experiments/lr_schedule/finetune_constant_lr*.sh, RUN_ORDER.md /
RUNNING.md, which otherwise leave it as a manual placeholder).

Counts via litdata's own StreamingDataset with TokensLoader(block_size=1):
each "item" is then exactly one token, so len(dataset) is the raw token
count with no remainder truncated by a larger block size. This reads each
chunk's local index.json metadata only -- it does not download/stream the
actual token bytes, so it's fast even for large datasets.

Usage:
    python count_tokens.py --data_path /path/to/tokenized_data/only-best
    python count_tokens.py --data_path /path/to/tokenized_data/only-best/train
"""
import argparse
from pathlib import Path

from litdata.streaming import StreamingDataset, TokensLoader


def count_split(split_dir: Path) -> int:
    dataset = StreamingDataset(
        input_dir=str(split_dir),
        item_loader=TokensLoader(block_size=1),
    )
    return len(dataset)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_path", required=True,
                     help="dataset dir -- either a single split (contains index.json directly) "
                          "or a parent dir with split subfolders (e.g. train/, valid/)")
    ap.add_argument("--splits", nargs="+", default=None,
                     help="which subfolder(s) of --data_path to count (default: autodetect "
                          "train/valid, or --data_path itself if it has no such subfolders)")
    args = ap.parse_args()

    data_path = Path(args.data_path)
    if not data_path.is_dir():
        raise SystemExit(f"error: not a directory: {data_path}")

    if args.splits:
        split_dirs = [(s, data_path / s) for s in args.splits]
    elif (data_path / "index.json").exists():
        split_dirs = [(data_path.name, data_path)]
    else:
        split_dirs = [(s, data_path / s) for s in ("train", "valid") if (data_path / s / "index.json").exists()]
        if not split_dirs:
            raise SystemExit(
                f"error: no index.json found directly under {data_path}, nor under a train/ or valid/ "
                f"subfolder -- pass --splits explicitly if this dataset uses different subfolder names"
            )

    total = 0
    for name, split_dir in split_dirs:
        if not (split_dir / "index.json").exists():
            print(f"[warn] skipping '{name}': no index.json under {split_dir}")
            continue
        n = count_split(split_dir)
        total += n
        print(f"{name}: {n:,} tokens")

    print(f"total: {total:,} tokens")


if __name__ == "__main__":
    main()
