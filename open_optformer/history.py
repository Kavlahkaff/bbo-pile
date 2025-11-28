import json
import random

from dataclasses import dataclass, field

import numpy as np
from syne_tune.config_space import Categorical, Float, Integer, Domain, config_space_from_json_dict, FiniteRange
from syne_tune.experiments import ExperimentResult


def preprocess(prompt: str):
    prompt = prompt.replace('parameter', "")
    prompt = prompt.replace('trial', "")
    prompt = prompt.replace('\"', "")
    prompt = prompt.replace(' ', "")
    return prompt


def quantize(x, x_min, x_max, q: int):
    """
    Quantize a value x to be in [0, q] based on the range [x_min, x_max].
    """
    if x_min == x_max:
        return 0
    x_norm = (x - x_min)/(x_max - x_min)
    return int(x_norm * q)


def dequantize(x, x_min, x_max, q: int):
    """
    Dequantize a value x from [0, q] to the range [x_min, x_max].
    """
    return x / q * (x_max - x_min) + x_min


def encode(x, hp: Domain, q: int, hp_name: str = ""):
    """
    Encode a value x based on the type of hyperparameter hp.
    """
    if isinstance(hp, Categorical):
       if hp_name == 'proc.skew_threshold' and np.isnan(x):
            x = 'None'
       if hp_name == 'proc.skew_threshold' and isinstance(x, float):
               x = str(x)
       return hp.categories.index(x)
    elif isinstance(hp, (Float, Integer, FiniteRange)):
        return quantize(x, hp.lower, hp.upper, q)
    else:
        raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")


@dataclass
class Trial:
    config: dict
    metric: int


@dataclass
class History:
    name: str
    algorithm: str
    config_space: dict
    num_numeric_tokens: int
    metric_names: list = field(default_factory=list)
    trials: list = field(default_factory=list)

    def add_trial(self, config, result):

        trial = Trial(config, result)
        self.trials.append(trial)

    def get_prompt(self, shuffle=False):
        string = f"benchmark:{self.name},algorithm:{self.algorithm},"

        # encode config-space
        hp_names = list(self.config_space.keys())

        if shuffle:
            random.shuffle(hp_names)
        for hp_name in hp_names:
            string += f"parameter:{self._encode_hp_config_space(hp_name)}"

        string += '&'

        # encode hyperparameters values
        if len(self.trials) > 0:
            for trial in self.trials:
                string += self._encode_trial_hp(trial, hp_names=hp_names)
        return string

    def _encode_trial_hp(self, trial: Trial, hp_names: list[str]) -> str:
        string = ""
        # TODO support quantile normalization
        y_values = [trial.metric for trial in self.trials]
        y_min = min(y_values)
        y_max = max(y_values)
        if y_min == y_max:
            y_max += 1  # Avoid division by zero in quantization
        for i, hp_name in enumerate(hp_names):
            hp = self.config_space[hp_name]
            if not isinstance(hp, Domain):
                continue
            if i > 0:
                string += ","
                hp_encoded = encode(x=trial.config[hp_name], hp=hp, hp_name=hp_name, q=self.num_numeric_tokens)
                string += str(hp_encoded)
            string += f"*"
            # TODO support other normalization
            string += f"{quantize(trial.metric, y_min, y_max, q=self.num_numeric_tokens)}"
            string += f"|"
        return string

    def _encode_hp_config_space(self, hp_name: str) -> str:
        string = "{"
        string += f"name:{hp_name},"
        hp = self.config_space[hp_name]
        if isinstance(hp, Categorical):
            string += f"type:CAT,"
            string += f"categories:{hp.categories},".replace(" ", "")
        elif isinstance(hp, Float):
            string += f"type:UNI,"
            string += f"min_value:{hp.lower},"
            string += f"max_value:{hp.upper},"
        elif isinstance(hp, Integer):
            string += f"type:INT,"
            string += f"min_value:{hp.lower},"
            string += f"max_value:{hp.upper},"
        elif isinstance(hp, FiniteRange):
            if hp.cast_int:
                string += f"type:INT,"
            else:
                string += f"type:UNI,"
            string += f"min_value:{hp.lower},"
            string += f"max_value:{hp.upper},"
        else:
            raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")
        string += "}"
        return string

    @classmethod
    def from_syne_tune_experiment(cls, experiment: ExperimentResult, max_num_trials: int = None):
        """
        Create a History object from a Syne Tune ExperimentResult.
        """
        metadata = experiment.metadata
        config_space = config_space_from_json_dict(json.loads(metadata['config_space']))
        metric_name = metadata["metric_names"][0]
        results = experiment.results

        benchmark_name = metadata['benchmark'] if 'benchmark' in metadata else metadata['entrypoint']
        algorithm_name = metadata['algorithm'] if 'algorithm' in metadata else metadata['scheduler_name']
        hist = cls(config_space=config_space,
                        name=benchmark_name,
                        algorithm=algorithm_name,
                        metric_names=metric_name)

        for i, (trial_id, trial) in enumerate(results.groupby('trial_id')):
            row = trial.iloc[-1]
            config = {k: row[f"config_{k}"] for k in config_space.keys()}
            result = row[metric_name]
            hist.add_trial(config, result)
            if i >= max_num_trials - 1 and max_num_trials is not None:
                break

        return hist