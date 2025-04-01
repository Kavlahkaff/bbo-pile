import argparse
import os
from pathlib import Path

import sentencepiece as spm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input_file",
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
        default=7000000,
    )
    args, _ = parser.parse_known_args()

    os.makedirs(args.output_path, exist_ok=True)
    spm.SentencePieceTrainer.Train(
        input=args.input_file,
        vocab_size=args.vocab_size,
        model_prefix=str(Path(args.output_path) / args.model_prefix),
        character_coverage=1.0,
        max_sentence_length=args.max_sentence_length,
        user_defined_symbols=['name', 'algorithm', 'benchmark', 'type',
                              'CAT', 'UNI', "|", "&", "*"] + [str(i) for i in range(1000)],
)