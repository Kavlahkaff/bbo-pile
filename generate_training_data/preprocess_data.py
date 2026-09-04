import argparse
import multiprocessing
import os
from functools import partial
from pathlib import Path

import litdata as ld
from litgpt import Tokenizer


def indexing(index, tokenizer):
    return tokenizer.encode(index)


def _num_available_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return multiprocessing.cpu_count()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default='./',
        help="",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='./',
        help="",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default='./',
        help="",
    )
    args, _ = parser.parse_known_args()

    os.makedirs(args.output_path, exist_ok=True)
    tokenizer = Tokenizer(args.tokenizer_dir)

    tokenize = partial(indexing, tokenizer=tokenizer)
    num_workers = _num_available_cores()
    print(f"Tokenizing with {num_workers} workers")

    for split in ['train', 'valid']:
        os.makedirs(Path(args.output_path) / split, exist_ok=True)
        filename = Path(args.input_path) / f'{split}.txt'
        file = open(filename)
        data = [line.rstrip() for line in file]
        file.close()
        print(split)
        print(f'N={len(data)}')
        print(f'sequence lengths={len(data[0])}')

        # The optimize function writes data in an optimized format.
        ld.optimize(
            fn=tokenize,                   # the function applied to each input
            inputs=data,           # the inputs to the function (here it's a list of numbers)
            output_dir= Path(args.output_path) / split,             # optimized data is stored here
            num_workers=num_workers,             # scale to the job's allocated CPUs
            chunk_bytes="64MB",                  # size of each chunk
            item_loader = ld.TokensLoader(block_size=534),
            mode='overwrite',
        )