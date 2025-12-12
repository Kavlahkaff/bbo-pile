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


def quantize(x, x_min, x_max, q=1000, log_scale=False):
    """
    Quantize a value x to be in [0, q] based on the range [x_min, x_max].
    """
    if x_min == x_max:
        return 0
    if log_scale:
        x = np.log(x + 1e-10)
        x_min = np.log(x_min + 1e-10)
        x_max = np.log(x_max + 1e-10)
    x_norm = (x - x_min)/(x_max - x_min)
    return int(x_norm * q)


def dequantize(x, x_min, x_max, q=1000, log_scale=False):
    """
    Dequantize a value x from [0, q] to the range [x_min, x_max].
    """
    if log_scale:
        x_min = np.log(x_min + 1e-10)
        x_max = np.log(x_max + 1e-10)
        return np.exp(x / q * (x_max - x_min) + x_min)
    return x / q * (x_max - x_min) + x_min


def encode(x, hp: Domain, hp_name: str = ""):
    """
    Encode a value x based on the type of hyperparameter hp.
    """
    if isinstance(hp, Categorical):
        #TODO: handle this in a more principled way
        if hp_name == 'proc.skew_threshold' and np.isnan(x):
            x = 'None'
        if hp_name == 'proc.skew_threshold' and isinstance(x, float):
               x = str(x)
        if hp_name == 'num_layers' and isinstance(x, np.int64):
               x = str(x)
        if hp_name == 'max_features':
           x = str(x)
        return f"<{hp.categories.index(x)}>"
    elif isinstance(hp, (Float, Integer, FiniteRange)):
        return quantize(x, hp.lower, hp.upper, log_scale=hp.log_scale)
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
    metric_names: list = field(default_factory=list)
    trials: list = field(default_factory=list)

    def add_trial(self, config, result):

        trial = Trial(config, result)
        self.trials.append(trial)

    def get_prompt(self, shuffle=False):
        string = f"benchmark:{self.name}\n"
        string += f"algorithm:{self.algorithm}\n"
        hypers = list(self.config_space.items())
        if shuffle:
            random.shuffle(hypers)
        # sort hyperparameters: continuous first, categorical last
        continues_hypers = []
        categorical_hypers = []
        for hp_name, hp in hypers:
            if isinstance(hp, Categorical):
                categorical_hypers.append((hp_name, hp))
            else:
                continues_hypers.append((hp_name, hp))
        hypers = continues_hypers + categorical_hypers
        string += f"search-space:\n"
        for hp_name, hp in hypers:
            string += "{"
            string += f"name:{hp_name},"

            if isinstance(hp, Categorical):

                string += f"type:CAT,"
                string += f"categories:{hp.categories}".replace(" ", "")
            elif isinstance(hp, Float):
                    string += f"type:UNI,"
                    string += f"min_value:{hp.lower},"
                    string += f"max_value:{hp.upper}"
            elif isinstance(hp, Integer):
                    string += f"type:INT,"
                    string += f"min_value:{hp.lower},"
                    string += f"max_value:{hp.upper}"
            elif isinstance(hp, FiniteRange):
                if hp.cast_int:
                    string += f"type:INT,"
                else:
                    string += f"type:UNI,"
                string += f"min_value:{hp.lower},"
                string += f"max_value:{hp.upper}"
            else:
                raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")
            string += "}\n"

        string += 'history\n'

        if len(self.trials) > 0:
            y_min = min(trial.metric for trial in self.trials)
            y_max = max(trial.metric for trial in self.trials)
            if y_min == y_max:
                y_max += 1  # Avoid division by zero in quantization
            for trial in self.trials:
                for i, (hp_name, hp) in enumerate(hypers):
                    if not isinstance(hp, Domain):
                        continue
                    if i > 0:
                        string += ","

                    hp_encoded = encode(trial.config[hp_name], hp, hp_name)
                    string += str(hp_encoded)
                string += f"*"

                string += f"{quantize(trial.metric, y_min, y_max)}"
                string += f"|"
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

if __name__ == "__main__":
    from syne_tune.config_space import uniform, randint, choice
    config_space = {
        'x': uniform(0, 1),
        'y': randint(0, 10),
        'z': choice(['a', 'b', 'c'])
    }
    history = History(name='test', algorithm='test', config_space=config_space)
    history.add_trial({'x': 0.5, 'y': 5, 'z': 'a'}, 0.5)
    history.add_trial({'x': 0.6, 'y': 6, 'z': 'b'}, 0.6)
    prompt = history.get_prompt()
    print(prompt)