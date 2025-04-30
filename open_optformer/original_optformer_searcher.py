import copy
import logging

import numpy as np
from typing import Optional, List, Dict, Any

from vizier.service import pyvizier as vz
from vizier import algorithms as vza
from optformer.t5x import inference_utils
from optformer.t5x import policies

from syne_tune.config_space import choice, Categorical
# from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import SingleObjectiveBaseSearcher

logger = logging.getLogger(__name__)

BBOB_INFERENCE_MODEL_KWARGS = {
    'checkpoint_path_or_model_dir': '/home/rakotoah/code/rakotoah-llmhpo/data/model_checkpoints/bbob/checkpoint_700000',
    'model_gin_file': '/home/rakotoah/code/rakotoah-llmhpo/code/optformer/optformer/t5x/configs/tasks/bbob.gin',
    'batch_size': 1,
}

class OriginalOptFormerSearcher(): # TODO: inherit from SingleObjectiveBaseSearcher
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
        config_space: Dict[str, Any],
        task_info: Dict = None,
        points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
        random_seed: int = None,
        designer_name: str = 'designer_hill_climb'
    ):

        problem = vz.ProblemStatement()
        for name, hp in config_space.items():
            print(f"Adding {name} to search space")
            if isinstance(hp, Categorical):
                problem.search_space.root.add_categorical_param(name, hp.categories)
            else:
                raise NotImplemented(f"Unsupported hyperparameter type {type(hp)} for {name}. ")

        problem.metric_information.append(vz.MetricInformation(name='error', goal=vz.ObjectiveMetricGoal.MAXIMIZE))

        inference_model = inference_utils.InferenceModel.from_checkpoint(
            **BBOB_INFERENCE_MODEL_KWARGS
        )
        self.model = policies.OptFormerDesigner(
            problem, inference_model=inference_model, designer_name=designer_name, temperature=0.9
        )

        self.history = []

        if task_info is None:
            self.task_info = {'name': "tst",
                              "algorithm": "optformer",
                              "metric_names": "error"}
        else:
            self.task_info = task_info

        self.history = []

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

        suggestion = self.model.suggest(1)
        self.history.append(suggestion[0])

        return suggestion[0].to_trial().parameters.as_dict()

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
        trial = self.history[trial_id].to_trial()
        trial.complete(vz.Measurement({'error': metric}))

        copied_trial = copy.deepcopy(trial)
        if self.model._metric_flipped:
            policies.flip_trial_metric_values(copied_trial)
        self.model._historical_study.trials.append(copied_trial)


    def on_trial_error(self, trial_id: int):
        """Called by scheduler if an evaluation job for a trial failed.

        The searcher should react appropriately (e.g., remove pending evaluations
        for this trial, not suggest the configuration again).

        :param trial_id: ID of trial whose evaluated failed
        """
        return

if __name__ == '__main__':

    import numpy as np

    config_space = {"a": choice([0, 1, 2, 3, 4])}

    searcher = OriginalOptFormerSearcher(config_space=config_space)

    for trial_id in range(5):
        config = searcher.suggest()
        print(config)
        metric = np.random.rand()
        searcher.on_trial_complete(trial_id, config, metric)
