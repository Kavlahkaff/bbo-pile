import os
import argparse
from pathlib import Path
import numpy as np
from syne_tune.config_space import choice
from src.open_optformer import Study


def random_trajectories(iterations, dimensionality):

    trajectory = np.random.randint(0, 2, size=(iterations, dimensionality))
    metric = np.sum(trajectory, axis=1)

    config_space = {f"x_{i}": choice([0, 1]) for i in range(dimensionality)}
    study = Study(config_space=config_space,
                       name="counting_ones",
                       algorithm="random_search",
                       metric_names=["error"],
                       )
    prompt = study.get_prompt()

    for i in range(iterations):
        Ti = trajectory[i]
        mi = metric[i]
        prompt += ','.join(str(tij) for tij in Ti) + '*' + str(mi) + '|'

    return prompt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='./',
        help="",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=60000,
        help="",
    )
    args, _ = parser.parse_known_args()

    num_trajectories = args.num_trajectories
    iterations = args.iterations
    dims = [5]

    d = Path(args.output_path) / 'data'
    os.makedirs(d, exist_ok=True)

    for mode in ['train', 'valid']:
        with open(d / f'counting_ones_{mode}.txt', 'w') as fh:
            for i in range(args.num_trajectories):
                dim = np.random.choice(dims)
                prompt = random_trajectories(iterations, dim)
                fh.write(prompt + "\n")