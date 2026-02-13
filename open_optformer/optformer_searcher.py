import os
import logging
import numpy as np
import torch

from typing import Optional, List, Dict, Any
from pathlib import Path


from litgpt.tokenizer import Tokenizer
from litgpt.generate.base import generate
from litgpt.model import GPT
from litgpt.config import Config
from open_optformer.history import History, dequantize
from transformers import AutoTokenizer, Qwen3ForCausalLM

from syne_tune.config_space import Integer, Float, FiniteRange, is_log_space
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import SingleObjectiveBaseSearcher
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)

logger = logging.getLogger(__name__)


class ConfigGrammar:
    """
    Generates a regex pattern to constrain LLM output to valid configurations.

    The output format is: {cont_values},{cat_values}*{output}|

    Example with 2 continuous and 1 categorical hyperparameter:
        "500,400,<0>*123|"

    Structure:
        - Continuous values: token IDs 0 to num_numeric_tokens-1, decoded to their string representation
        - Categorical values: tokens <0>, <1>, ..., <num_categorical_tokens-1>
        - All hyperparameter values are comma-separated
        - '*' separates hyperparameters from the predicted output
        - '|' marks the end of the sequence

    The regex is built using actual token strings from the tokenizer vocabulary,
    ensuring the model only generates valid token sequences.
    """

    def __init__(
        self,
        tokenizer,
        config_space,
        n_continuous: int,
        n_categorical: int,
        hp_cat_names: list[str],
        num_numeric_tokens: int = 1000,
        num_categorical_tokens: int = 15,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer with convert_ids_to_tokens method
            n_continuous: Number of continuous hyperparameters
            n_categorical: Number of categorical hyperparameters
            num_numeric_tokens: Number of quantization levels for continuous values (token IDs 0 to num_numeric_tokens-1)
            num_categorical_tokens: Maximum number of categories (tokens <0> to <num_categorical_tokens-1>)
        """
        self.tokenizer = tokenizer
        self.n_continuous = n_continuous
        self.n_categorical = n_categorical
        self.hp_cat_names = hp_cat_names
        self.config_space = config_space
        self.num_numeric_tokens = num_numeric_tokens
        self.num_categorical_tokens = num_categorical_tokens

    def _get_continuous_tokens(self) -> list[str]:
        """
        Get string representations of valid continuous value tokens.

        Token IDs 0 to num_numeric_tokens-1 are used directly as quantized values.
        Returns their string representations from the tokenizer vocabulary.
        """
        return [
            self.tokenizer.convert_ids_to_tokens(i)
            for i in range(self.num_numeric_tokens)
        ]

    def _get_categorical_tokens(self) -> list[str]:
        """
        Get string representations of valid categorical tokens.

        Categorical values are encoded as <0>, <1>, ..., <num_categorical_tokens-1>.
        """
        return [f'<{i}>' for i in range(self.num_categorical_tokens)]

    def _get_separator_tokens(self) -> dict[str, str]:
        """
        Get string representations of separator tokens.

        Returns dict with keys: 'comma', 'star', 'pipe'
        """
        token_to_id = self.tokenizer.convert_tokens_to_ids
        return {
            'comma': self.tokenizer.convert_ids_to_tokens(token_to_id(',')),
            'star': self.tokenizer.convert_ids_to_tokens(token_to_id('*')),
            'pipe': self.tokenizer.convert_ids_to_tokens(token_to_id('|')),
        }

    def _escape_regex(self, s: str) -> str:
        """Escape special regex characters in a string."""
        import re
        return re.escape(s)

    def _build_continuous_pattern(self) -> str:
        """
        Build regex pattern matching any valid continuous token.

        Returns alternation of all valid continuous token strings.
        """
        tokens = self._get_continuous_tokens()
        escaped = [self._escape_regex(t) for t in tokens]
        return '(' + '|'.join(escaped) + ')'

    def _build_categorical_pattern(self, max_categories: int = None) -> str:
        """
        Build regex pattern matching any valid categorical token.

        Args:
            max_categories: If provided, only allow tokens <0> to <max_categories-1>.
                           If None, allows all categorical tokens.

        Returns alternation of all valid categorical token strings.
        """
        if max_categories is None:
            tokens = self._get_categorical_tokens()
        else:
            tokens = [f'<{i}>' for i in range(max_categories)]
        escaped = [self._escape_regex(t) for t in tokens]
        return '(' + '|'.join(escaped) + ')'

    def build_regex(self) -> str:
        """
        Build the complete regex pattern for valid configuration strings.

        Uses actual token strings from the tokenizer to ensure the model
        only generates valid token sequences.

        Returns:
            Regex pattern string for guided decoding
        """
        # TODO important note, right now we constrain the model to predict a token among the 1000 options
        #  it would be more efficient to check if values are in a range as values are continuous
        cont_pattern = self._build_continuous_pattern()
        separators = self._get_separator_tokens()

        comma = self._escape_regex(separators['comma'])
        star = self._escape_regex(separators['star'])
        pipe = self._escape_regex(separators['pipe'])

        # Build list of patterns for each hyperparameter
        patterns = []

        # Add patterns for continuous hyperparameters (come first)
        for _ in range(self.n_continuous):
            patterns.append(cont_pattern)

        # Add patterns for categorical hyperparameters (come after continuous)
        # Each categorical HP is restricted to only its valid category tokens
        for hp_cat in self.hp_cat_names:
            n_categories = len(self.config_space[hp_cat].categories)
            patterns.append(self._build_categorical_pattern(n_categories))

        # Join all hyperparameter patterns with comma separator
        if patterns:
            hp_pattern = comma.join(patterns)
            # Full pattern: {hp_values}*{output}|
            # Output is also a continuous value (predicted metric)
            regex = hp_pattern + star + cont_pattern + pipe
        else:
            # Edge case: no hyperparameters, just output
            regex = star + cont_pattern + pipe

        return regex


def detect_hf_checkpoint(path):
    """
    Returns: bool: True if the checkpoint is a hf model, False otherwise
    """
    path = os.fspath(path)

    files = set(os.listdir(path))

    lit_markers = {"lit_model.pth", "hyperparameters.yaml", "model_config.yaml"}
    lit_hits = lit_markers & files
    if lit_hits:
        logging.debug("found litgpt model")
        return False

    hf_model_files = [f for f in files if f.endswith(".safetensors")]
    if "config.json" in files and hf_model_files:
        logging.debug("found hf model")
        return True



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
        n_sample_configurations: int = 1,
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
                task_info=task_info,
                n_sample_configurations=n_sample_configurations,
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
        remove_names: bool = True,
        n_sample_configurations: int = 1,
        use_vllm: bool = True,
    ):
        """
        :param checkpoint_dir:
        :param config_space:
        :param task_info:
        :param points_to_evaluate:
        :param random_seed:
        :param num_numeric_tokens:
        :param num_categorical_tokens:
        :param n_sample_configurations: number of configurations to sample, pick the one with best predicted performance.
        """
        super().__init__(config_space, points_to_evaluate, random_seed)
        if random_seed is not None:
            torch.random.manual_seed(random_seed)
        self.use_hf_checkpoint = detect_hf_checkpoint(checkpoint_dir)
        self.use_vllm = use_vllm
        if self.use_vllm:
            assert self.use_hf_checkpoint, "Can only use vllm with a HF checkpoint, convert the litgpt checkpoint first."
        if self.use_vllm:
            from vllm import LLM
            from vllm.config.structured_outputs import StructuredOutputsConfig
            self.model = LLM(model=str(checkpoint_dir), structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"))
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.use_hf_checkpoint:
            self.model = Qwen3ForCausalLM.from_pretrained(checkpoint_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            config = Config.from_file(str(checkpoint_dir / 'model_config.yaml'))
            self.model = GPT(config)
            self.tokenizer = Tokenizer(str(checkpoint_dir))
            state_dict = torch.load(
                str(checkpoint_dir / 'lit_model.pth'),
                weights_only=True,
                map_location=torch.device('cpu') if not torch.cuda.is_available() else None
            )
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.model.load_state_dict(state_dict)
        self.random_state = np.random.RandomState(random_seed)
        self.num_numeric_tokens = num_numeric_tokens
        self.num_categorical_tokens = num_categorical_tokens
        self.n_sample_configurations = n_sample_configurations
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
                             remove_names=remove_names,
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
            configs, ys = self._sample_n_configs()
            # return best of n
            if len(configs) == 0:
                logging.warning('Sampling failed, return a random configuration!')
                return {k: v.sample() for k, v in self.config_space.items()}
            print(f'valid config: {len(configs)}/{self.n_sample_configurations}')
            return configs[np.argmin(ys)]

    def _sample_n_configs(self):

        configs = []
        ys = []
        prompt = self.study.get_prompt()
        completions = self._generate_n_suggestions(prompt=prompt)

        for completion in completions:
            try:
                # decode configuration, if possible
                config, y = self._decode_config(completion)

                # add constant hyperparameters
                for k, v in self.config_space.items():
                    if not hasattr(v, "sample"):
                        config[k] = v
                configs.append(config)
                ys.append(y)

            except ValueError as e:
                logging.warning(f"Could not sample because of error: {str(e)}, skipping sampled configuration.")

        return configs, ys

    def _generate_n_suggestions(self, prompt: str) -> List[List[int]]:
        "Generate a string like `500,400,<0>*123|`"
        if self.use_hf_checkpoint:
            if self.use_vllm:
                # 500,400,<0>*123|
                max_new_tokens = (len(self.hp_cont_names) + len(self.hp_cat_names)) * 2 + 1

                # Build regex grammar to constrain output to valid configurations
                grammar = ConfigGrammar(
                    tokenizer=self.tokenizer,
                    config_space=self.config_space,
                    n_continuous=len(self.hp_cont_names),
                    n_categorical=len(self.hp_cat_names),
                    hp_cat_names=self.hp_cat_names,
                    num_numeric_tokens=self.num_numeric_tokens,
                    num_categorical_tokens=self.num_categorical_tokens,
                )
                regex_pattern = grammar.build_regex()

                from vllm import SamplingParams
                from vllm.sampling_params import StructuredOutputsParams
                sampling_params = SamplingParams(
                    max_tokens=max_new_tokens,
                    n=self.n_sample_configurations,
                    structured_outputs=StructuredOutputsParams(regex=regex_pattern),
                )
                outputs = self.model.generate([prompt], sampling_params)
                tokens_configs = [list(output.token_ids) for output in outputs[0].outputs]
            else:
                with torch.no_grad():
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    prompt_length = inputs['input_ids'].shape[1]

                    # number of tokens to return including the prompt is 2 per hyperparameters counting for the comma
                    # minus one as there is trailing comma, plus two for the * and token of the predicted output
                    max_new_tokens = (len(self.hp_cont_names) + len(self.hp_cat_names)) * 2 + 1
                    eos_token_id = self.tokenizer.convert_tokens_to_ids("|")

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=self.n_sample_configurations,
                        do_sample=True,
                        eos_token_id=eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                    # Remove prompt from the beginning of each sequence
                    tokens_configs = [output[prompt_length:].tolist() for output in outputs]
        else:
            with torch.no_grad():

                prompt_tokens = self.tokenizer.encode(prompt)[-self.model.max_seq_length:]
                self.model.set_kv_cache(batch_size=1)

                # number of tokens to return including the prompt is 2 per hyperparameters counting for the comma
                # minus one as there is trailing comma, plus two for the * and token of the predicted output
                max_returned_tokens = len(prompt_tokens) + (len(self.hp_cont_names) + len(self.hp_cat_names)) * 2 + 1

                # enables parallelism
                # from pyparfor import parfor
                # tokens_configs = parfor(
                #     lambda i: generate(
                #         model=self.model,
                #         prompt=prompt_tokens,
                #         max_returned_tokens=max_returned_tokens,
                #         include_prompt=False,
                #         eos_id=self.tokenizer.token_to_id("|"),
                #     ).tolist(),
                #     list(range(self.n_sample_configurations)),
                #     engine="futures",
                # )

                tokens_configs = [
                    generate(
                        model=self.model,
                        prompt=prompt_tokens,
                        max_returned_tokens=max_returned_tokens,
                        include_prompt=False,
                        eos_id=self.tokenizer.token_to_id("|"),
                    ).tolist()
                    for _ in range(self.n_sample_configurations)
                ]
        return tokens_configs

    def _decode_config(self, tokens_config: list[int]) -> tuple[Dict[str, Any], float]:
        # decode configuration in the form of "500,500,<0>*0|"
        token_to_id = self.tokenizer.convert_tokens_to_ids if self.use_hf_checkpoint else self.tokenizer.token_to_id

        star_index = tokens_config.index(token_to_id("*"))

        if star_index >= len(tokens_config) - 1:
            raise ValueError(f"Star index {star_index} is out of bounds.")
        tokens_hps = tokens_config[:star_index]
        token_output = tokens_config[star_index + 1]

        # we decode the tokens into a configuration dictionary
        config = {}

        hp_value_tokens = [x for x in tokens_hps if x != token_to_id(",")]

        if len(hp_value_tokens) != len(self.hp_cont_names) + len(self.hp_cat_names):
            logging.warning("wrong length")

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
                    token_to_id(f"<{i}>"): cat
                    for i, cat in enumerate(self.config_space[hp_name].categories)
                }

                if hp_token not in tokens_per_category:
                    # TODO should we rather fail in this case? How frequently does this happen?
                    # can be fixed if using HF interface as it allows to restrict the tokens that can be sampled
                    logging.warning(f"Could not read category {hp_name}, got token {hp_token}.")
                    config[hp_name] = self.config_space[hp_name].sample(random_state=self.random_state)
                else:
                    config[hp_name] = tokens_per_category[hp_token]

        for hp_name in self.hp_cat_names:
            if hp_name not in config:
                logging.warning(f"Did not sample category {hp_name}, sampling randomly")
                config[hp_name] = self.config_space[hp_name].sample(random_state=self.random_state)

        # note we return token_output as the predicted output, we could also apply the invert quantization but it does
        # not matter as it is a monotonic operation and we are only interested in picking the lowest predicted output
        return config, token_output

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

    checkpoint_dir = pathlib.Path(__file__).parent.parent / "models" / "checkpoint"
    searcher = OptFormerSearcher(config_space=config_space, checkpoint_dir=checkpoint_dir)

    for trial_id in range(20):
        config = searcher.suggest()
        print(config)
        metric = np.random.rand()
        searcher.on_trial_complete(trial_id, config, metric)
