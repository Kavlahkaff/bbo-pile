from argparse import ArgumentParser

from syne_tune import Reporter
import numpy as np


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--x0",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x1",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x2",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x3",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x4",
        type=int,
        required=False,
        default=0,
    )

    parser.add_argument(
        "--x5",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x6",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x7",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x8",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x9",
        type=int,
        required=False,
        default=0,
    )
    args, _ = parser.parse_known_args()
    args = vars(args)
    reporter = Reporter()
    config = [args[f'x_{i}'] for i in range(10)]
    ones = np.sum(config)

    reporter(feval=ones)
