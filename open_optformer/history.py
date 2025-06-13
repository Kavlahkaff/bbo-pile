from dataclasses import dataclass, field

from syne_tune.config_space import Categorical, Float, Integer, Domain


@dataclass
class Trial:
    config: dict
    metric: int


@dataclass
class Study:
    name: str
    algorithm: str
    config_space: dict
    metric_names: list = field(default_factory=list)
    trials: list = field(default_factory=list)

    def add_trial(self, config, result):

        trial = Trial(config, result)
        self.trials.append(trial)

    def get_prompt(self):
        string = f"benchmark:{self.name},"
        string += f"algorithm:{self.algorithm},"

        for hp_name, hp in self.config_space.items():
            string += f"parameter:"
            string += "{"
            string += f"name:{hp_name}, "

            if isinstance(hp, Categorical):

                string += f"type:CAT,"
                string += f"categories:{hp.categories},"
            elif isinstance(hp, Float):
                    string += f"type:UNI,"
                    string += f"min_value:{hp.lower},"
                    string += f"max_value:{hp.upper},"
            elif isinstance(hp, Integer):
                    string += f"type:Int,"
                    string += f"min_value:{hp.lower},"
                    string += f"max_value:{hp.upper},"
            string += "}"

        string += '&'
        for trial in self.trials:
#            string += "trial:{"
            for i, hp in enumerate(self.config_space):
                if not isinstance(hp, Domain):
                    continue
                if i > 0:
                        string += ","
                string += str(trial.config[hp])
            string += f"*"
            string += f"{trial.metric}"
            string += f"|"
 #           string += "},"
        return string
