from argparse import ArgumentParser

from syne_tune import Reporter
import numpy as np


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--x_0",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x_1",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x_2",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x_3",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x_4",
        type=int,
        required=False,
        default=0,
    )

    parser.add_argument(
        "--x_5",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x_6",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x_7",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x_8",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x_9",
        type=int,
        required=False,
        default=0,
    )

    parser.add_argument(
        "--x_10",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x_11",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x_12",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x_13",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x_14",
        type=int,
        required=False,
        default=0,
    )

    parser.add_argument(
        "--x_15",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x_16",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x_17",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x_18",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--x_19",
        type=int,
        required=False,
        default=0,
    )
    args, _ = parser.parse_known_args()
    args = vars(args)
    reporter = Reporter()
    config = [args[f'x_{i}'] for i in range(20)]
    ones = np.sum(config)

    reporter(feval=ones)
