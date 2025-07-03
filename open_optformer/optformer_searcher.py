import logging
from itertools import takewhile

import torch
import numpy as np

from typing import Optional, List, Dict, Any
from pathlib import Path


from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from litgpt.model import GPT

from open_optformer.history import Study

from syne_tune.config_space import Integer, Categorical, Float
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import SingleObjectiveBaseSearcher
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)

logger = logging.getLogger(__name__)


def preprocess(prompt: str):
    prompt = prompt.replace('parameter', "")
    prompt = prompt.replace('trial', "")
    prompt = prompt.replace('\"', "")
    prompt = prompt.replace(' ', "")
    return prompt



def select_token(logits, pos):
    #m = logits[:, pos].argmax(dim=-1)
    probs = torch.nn.functional.softmax(logits[:, pos], dim=-1).detach().numpy()[0, :]
    m = np.random.choice(np.arange(pos.shape[0]), p=probs)
    token = pos[m]
    return token

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
    ):
        super().__init__(config_space, points_to_evaluate, random_seed)

        config = Config.from_file(str(checkpoint_dir / 'model_config.yaml'))
        self.model = GPT(config)

#        self.tokenizer = Tokenizer(str(Path(__file__).parent / "data" / "tokenizer"))
        self.tokenizer = Tokenizer(str(checkpoint_dir))
        self.model.load_state_dict(torch.load(str(checkpoint_dir / 'lit_model.pth'), weights_only=True)['model'])
        self.history = []

        if task_info is None:
            self.task_info = {'name': "tst",
                              "algorithm": "BORE",
                              "metric_names": "error"}
        else:
            self.task_info = task_info

        self.study = Study(config_space=config_space,
                           name=self.task_info['name'],
                           algorithm=self.task_info['algorithm'],
                           metric_names=[self.task_info['metric_names']],
        )

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

        prompt = preprocess(self.study.get_prompt())
        token = self.tokenizer.encode(prompt)
        prompt_size = token.size(0)
        input_pos = torch.arange(0, token.size(0))
        input_pos_maxp1 = torch.tensor([prompt_size])

        self.model.set_kv_cache(batch_size=1)
        prefill_token = True
        config = {}

        for hp_name, hp in self.config_space.items():
            logits = self.model(token.view(1, -1), input_pos, input_pos_maxp1=input_pos_maxp1)[:, -1]

            if isinstance(hp, Float) or isinstance(hp, Integer):
                # pick value in [0, Q] with the highest probability
                idx = torch.tensor([self.tokenizer.encode(str(i))[-1] for i in range(1000)], dtype=torch.int)
                token = select_token(logits, idx)
                config[hp_name] = token / 1000 * (hp.upper - hp.lower) + hp.lower

            elif isinstance(hp, Categorical):
                #  pick the category with the highest probability
                idx = torch.tensor([self.tokenizer.encode(str(cat))[-1] for cat in hp.categories], dtype=torch.int)
                token = select_token(logits, idx)
                value = int(self.tokenizer.decode(token))
                config[hp_name] = hp.categories[value]
            if prefill_token:
                prefill_token = False
                input_pos = torch.tensor([prompt_size],dtype=torch.int64)
            else:
                input_pos.add_(1)
            input_pos_maxp1.add_(1)
            self.model(token.view(1, -1), input_pos, input_pos_maxp1=input_pos_maxp1)
            input_pos.add_(1)
            token = torch.tensor([1012])
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
        self.study.add_trial(config, metric)


    def on_trial_error(self, trial_id: int):
        """Called by scheduler if an evaluation job for a trial failed.

        The searcher should react appropriately (e.g., remove pending evaluations
        for this trial, not suggest the configuration again).

        :param trial_id: ID of trial whose evaluated failed
        """
        return

if __name__ == '__main__':

    import numpy as np
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
