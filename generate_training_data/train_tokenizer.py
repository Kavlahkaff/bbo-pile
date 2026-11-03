import argparse
import os
from pathlib import Path

import sentencepiece as spm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input_folder",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        type=str,
    )
    parser.add_argument(
        "--model_prefix",
        default='tokenizer',
        type=str,
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=1040
    )
    parser.add_argument(
        "--max_sentence_length",
        type=int,
        default=6000,
    )
    parser.add_argument(
        "--input_sentence_size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        default=None,
        help="text files to train on (alternative to --input_folder, which uses train.txt and valid.txt)",
    )
    parser.add_argument(
        "--extra_symbols",
        type=str,
        default=None,
        help="file with one extra user-defined symbol per line (e.g. optimizer tags), each becomes one token",
    )
    args, _ = parser.parse_known_args()

    special_token = ['name', 'algorithm', 'benchmark', 'type', 'history',
     'CAT', 'UNI', 'INT', "|", "&", "*", ",", "log_scale", "linear_scale"]

    # Float, integer parameters
    special_token += [str(i) for i in range(1000)]
    # Categorical parameters
    special_token += [f"<{i}>" for i in range(15)]

    if args.extra_symbols is not None:
        extra = [l.strip() for l in open(args.extra_symbols) if l.strip() and not l.startswith("#")]
        special_token += [t for t in extra if t not in special_token]

    if args.input_files is not None:
        inputs = ",".join(args.input_files)
    else:
        input_folder = Path(args.input_folder)
        inputs = f'{input_folder / "train.txt"},{input_folder / "valid.txt"}'
    os.makedirs(args.output_path, exist_ok=True)
    spm.SentencePieceTrainer.Train(
        input=inputs,
        vocab_size=args.vocab_size,
        model_prefix=str(Path(args.output_path) / args.model_prefix),
        character_coverage=1.0,
        input_sentence_size=args.input_sentence_size,
        max_sentence_length=args.max_sentence_length,
        shuffle_input_sentence=True,
        user_defined_symbols=special_token,
)