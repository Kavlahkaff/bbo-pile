import logging
import numpy as np
import torch
import torch.nn.functional as F

from typing import Optional, List, Dict, Any
from pathlib import Path


from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from litgpt.model import GPT
from litgpt.generate.base import generate

from open_optformer.history import History, dequantize

from syne_tune.config_space import Integer, Categorical, Float, FiniteRange, is_log_space
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import SingleObjectiveBaseSearcher
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)

logger = logging.getLogger(__name__)


class OptformerScheduler(SingleObjectiveScheduler):
    """
   """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        checkpoint_dir: Path,
        task_info: Dict = None,
        do_minimize: Optional[bool] = True,
        random_seed: Optional[int] = None,
        points_to_evaluate: Optional[List[dict]] = None,
    ):
        super(OptformerScheduler, self).__init__(
            config_space=config_space,
            metric=metric,
            do_minimize=do_minimize,
            searcher=OptFormerSearcher(
                config_space=config_space,
                points_to_evaluate=points_to_evaluate,
                random_seed=random_seed,
                checkpoint_dir=checkpoint_dir,
                task_info=task_info
            ),
            random_seed=random_seed,
        )




class OptFormerSearcher(SingleObjectiveBaseSearcher):
    """

    :param config_space: Configuration space
    :param points_to_evaluate: List of configurations to be evaluated
        initially (in that order). Each config in the list can be partially
        specified, or even be an empty dict. For each hyperparameter not
        specified, the default value is determined using a midpoint heuristic.
        If ``None`` (default), this is mapped to ``[dict()]``, a single default config
        determined by the midpoint heuristic. If ``[]`` (empty list), no initial
        configurations are specified.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        config_space: Dict[str, Any],
        task_info: Dict = None,
        points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
        random_seed: int = None,
        num_numeric_tokens: int = 1000,
        num_categorical_tokens: int = 15,
    ):
        super().__init__(config_space, points_to_evaluate, random_seed)
        if random_seed is not None:
            torch.random.manual_seed(random_seed)
        config = Config.from_file(str(checkpoint_dir / 'model_config.yaml'))
        self.model = GPT(config)
        self.random_state = np.random.RandomState(random_seed)
        self.num_numeric_tokens = num_numeric_tokens
        self.num_categorical_tokens = num_categorical_tokens
        self.tokenizer = Tokenizer(str(checkpoint_dir))
        state_dict = torch.load(
            str(checkpoint_dir / 'lit_model.pth'),
            weights_only=True,
            map_location=torch.device('cpu') if not torch.cuda.is_available() else None
        )
        if 'model' in state_dict:
            state_dict = state_dict['model']
        self.model.load_state_dict(state_dict)
        self.history = []

        if task_info is None:
            self.task_info = {'name': "tst",
                              "algorithm": "BORE",
                              "metric_names": "error"}
        else:
            self.task_info = task_info

        self.study = History(config_space=config_space,
                             name=self.task_info['name'],
                             algorithm=self.task_info['algorithm'],
                             metric_names=[self.task_info['metric_names']],
                             num_numeric_tokens=self.num_numeric_tokens,
                             )

        # Sort hp to have continuous first and categorical after
        self.hp_cont_names = [
            hp_name
            for hp_name, hp in config_space.items()
            if isinstance(hp, (Float, Integer, FiniteRange))
        ]
        self.hp_cat_names = [
            hp_name
            for hp_name, hp in config_space.items()
            if not isinstance(hp, (Float, Integer, FiniteRange))
        ]
        self.config_space = {
            k: self.config_space[k]
            for k in self.hp_cont_names + self.hp_cat_names
        }



    def suggest(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Suggest a new configuration.

        Note: Query :meth:`_next_initial_config` for initial configs to return
        first.

        :param kwargs: Extra information may be passed from scheduler to
            searcher
        :return: New configuration. The searcher may return None if a new
            configuration cannot be suggested. In this case, the tuning will
            stop. This happens if searchers never suggest the same config more
            than once, and all configs in the (finite) search space are
            exhausted.

        """
        config = self._next_points_to_evaluate()
        if config is not None:
            return config
        else:

            # generate tokens of the configuration
            prompt = self.study.get_prompt()
            prompt_tokens = self.tokenizer.encode(prompt)[-self.model.max_seq_length:]
            self.model.set_kv_cache(batch_size=1)

            with torch.no_grad():
                # number of tokens to return including the prompt is 2 per hyperparameters counting for the comma
                # minus one as there is trailing comma
                max_returned_tokens = len(prompt_tokens) + (len(self.hp_cont_names) + len(self.hp_cat_names)) * 2 - 1
                tokens_hps = generate(
                    model=self.model,
                    prompt=prompt_tokens,
                    max_returned_tokens=max_returned_tokens,
                    include_prompt=False,
                )

            # decode the tokens of the configuration, if possible
            try:
                config = self._decode_config(tokens_hps.tolist())

                # add constant hyperparameters
                for k, v in config_space.items():
                    if not hasattr(v, "sample"):
                        config[k] = v

            except ValueError as e:
                # shows the error draw a random config
                print(f"Could not sample because of error: {str(e)}, returning random config.")
                config = {
                    k: v.sample(random_state=self.random_state) if hasattr(v, "sample") else v
                    for k, v in config_space.items()
                }
            finally:
                return config

    def _decode_config(self, tokens_hps: list[int]) -> Dict[str, Any]:
        # we decode the tokens into a configuration dictionary
        config = {}
        hp_value_tokens = [x for x in tokens_hps if x != self.tokenizer.token_to_id(",")]
        if len(hp_value_tokens) != len(self.hp_cont_names) + len(self.hp_cat_names):
            print("wrong length")

        for i, (hp_name, hp_token) in enumerate(zip(self.hp_cont_names + self.hp_cat_names, hp_value_tokens)):
            is_continuous_hp = i < len(self.hp_cont_names)
            if is_continuous_hp:
                config[hp_name] = dequantize(
                    x=hp_token,
                    x_min=self.config_space[hp_name].lower,
                    x_max=self.config_space[hp_name].upper,
                    q=self.num_numeric_tokens,
                    log_scale=is_log_space(self.config_space[hp_name]),
                )
            else:
                # categorical
                tokens_per_category = {
                    self.tokenizer.encode(f"<{i}>").tolist()[1]: cat
                    for i, cat in enumerate(self.config_space[hp_name].categories)
                }
                if hp_token not in tokens_per_category:
                    # TODO should we rather fail in this case? How frequently does this happen?
                    # can be fixed if using HF interface as it allows to restrict the tokens that can be sampled
                    print(f"Could not read category {hp_name}, got token {hp_token}.")
                    config[hp_name] = self.random_state.choice(self.config_space[hp_name].categories)
                else:
                    config[hp_name] = tokens_per_category[hp_token]

        for hp_name in self.hp_cat_names:
            if hp_name not in config:
                print(f"Did not sample category {hp_name}, sampling randomly")
                config[hp_name] = self.random_state.choice(self.config_space[hp_name].categories)

        return config

    def on_trial_complete(
            self,
            trial_id: int,
            config: Dict[str, Any],
            metric: float,
            resource_level: int = None,
    ):
        """Inform searcher about result

        The scheduler passes every result. If ``update == True``, the searcher
        should update its surrogate model (if any), otherwise ``result`` is an
        intermediate result not modelled.

        The default implementation calls :meth:`_update` if ``update == True``.
        It can be overwritten by searchers which also react to intermediate
        results.

        :param trial_id: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param config: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param metric: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        """
        if isinstance(metric, list):
            self.study.add_trial(config, metric[0])
        else:
            self.study.add_trial(config, metric)

    def on_trial_error(self, trial_id: int):
        """Called by scheduler if an evaluation job for a trial failed.

        The searcher should react appropriately (e.g., remove pending evaluations
        for this trial, not suggest the configuration again).

        :param trial_id: ID of trial whose evaluated failed
        """
        return

if __name__ == '__main__':

    import pathlib
    from syne_tune.config_space import randint, choice

    config_space = {"a": choice([0, 1, 2, 3, 4])}

    checkpoint_dir = pathlib.Path(__file__).parent / "models" / "small_custom_model" / "step-00000800"
    searcher = OptFormerSearcher(config_space=config_space, checkpoint_dir=checkpoint_dir)

    trial_id = 0
    config = searcher.suggest()
    print(config)
    metric = np.random.rand()
    searcher.on_trial_complete(trial_id, config, metric)
