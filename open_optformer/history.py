import json
import random

from dataclasses import dataclass, field

import numpy as np
from syne_tune.config_space import Categorical, Float, Integer, Domain, config_space_from_json_dict, FiniteRange, is_log_space
from syne_tune.experiments import ExperimentResult


def quantize(x, x_min, x_max, q=1000, log_scale=False):
    """
    Quantize a value x to be in [0, q-1] based on the range [x_min, x_max].
    q is the number of quantization levels.
    """
    if x_min == x_max:
        return 0
    if log_scale:
        x = np.log(x + 1e-10)
        x_min = np.log(x_min + 1e-10)
        x_max = np.log(x_max + 1e-10)
    x_norm = (x - x_min)/(x_max - x_min)
    return int(round(x_norm * (q - 1)))


def dequantize(x, x_min, x_max, q=1000, log_scale=False):
    """
    Dequantize a value x from [0, q-1] to the range [x_min, x_max].
    q is the number of quantization levels.
    """
    if log_scale:
        x_min = np.log(x_min + 1e-10)
        x_max = np.log(x_max + 1e-10)
        return np.exp(x / (q - 1) * (x_max - x_min) + x_min)
    return x / (q - 1) * (x_max - x_min) + x_min


def symlog(x):
    """Signed log1p: sign(x) * log(1 + |x|). Monotonic and defined for any
    real x (unlike `quantize`'s `log_scale`, which assumes x > 0 via
    `log(x + 1e-10)`) -- used for `History.metric_log_scale`, since metric
    values are frequently negative (e.g. a `max`-mode objective like accuracy
    gets sign-flipped to always-minimize, see `from_syne_tune_experiment`) or
    span zero. Compresses large magnitudes so that objectives whose scale
    varies multiplicatively across configs/tasks (typical of regression
    losses like MSE/RMSE, unlike a metric bounded in e.g. [0, 1]) don't have
    their quantization resolution dominated by the largest few values.
    """
    return np.sign(x) * np.log1p(np.abs(x))


def encode(x, hp: Domain, q: int = 1000, hp_name: str = ""):
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
        return quantize(x, hp.lower, hp.upper, q, log_scale=is_log_space(hp))
    else:
        raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")


@dataclass
class Trial:
    config: dict
    metric: int
    advantage: float = None


@dataclass
class History:
    name: str
    algorithm: str
    config_space: dict
    num_numeric_tokens: int = 1000
    metric_names: list = field(default_factory=list)
    trials: list = field(default_factory=list)
    remove_names: bool = False
    # Selects how the (y_min, y_max) metric quantization range is computed
    # per History (ignored when `metric_range_override` is set -- that
    # always wins). "original" (default) matches the upstream bbo-pile
    # repo: a single GLOBAL range spanning every trial in this History
    # (including ones after the trial being quantized), no percentile
    # clipping. "causal" is the opt-in quantization-causal-robust fix: the
    # range for trial t is computed only from trials observed up to and
    # including t (never from later trials, matching what is available at
    # inference time), with the bad tail percentile-clipped by
    # `metric_percentile_hi`/`metric_percentile_lo` below to stop a single
    # diverged trial from consuming the token resolution. See
    # _metric_ranges(). "quantile" drops the (y_min, y_max) range entirely:
    # each trial's token is its rank among ALL trials of this History
    # (average rank for ties), scaled to [0, num_numeric_tokens-1], so the
    # tokens are uniform per document, immune to a single diverged trial and
    # need no oracle range at inference (the prompt is rebuilt from all
    # observed trials on every suggest()). Incompatible with
    # `metric_range_override`; `metric_log_scale` is a no-op under ranks.
    # See _metric_tokens().
    metric_quantization_mode: str = "original"
    # Percentile (0-100) used to clip the "bad" tail of observed metrics
    # (metrics are already normalized so that lower == better, see
    # from_syne_tune_experiment). Only used when metric_quantization_mode ==
    # "causal". Defaults to 95 to protect against divergent/outlier trials
    # (e.g. fcnet) stealing quantization resolution.
    metric_percentile_hi: float = 95.0
    # Percentile used to clip the "good" tail. Only used when
    # metric_quantization_mode == "causal". Defaults to 0 (true min, no
    # clipping) since resolution among the best trials matters most for search.
    metric_percentile_lo: float = 0.0
    # Debug/ablation knob: when set to (y_min, y_max), every trial is
    # quantized against this FIXED range instead of metric_quantization_mode
    # above -- e.g. the task's true (oracle) metric range, to test whether
    # quantization mismatch (not model capability) explains a gap to
    # baselines. None (default) leaves metric_quantization_mode in effect.
    # See _metric_ranges().
    metric_range_override: tuple = None
    # Quantize the metric on a signed-log scale (see `symlog`) instead of
    # linearly. Off by default -- a metric already bounded on a fixed,
    # roughly-additive scale (e.g. accuracy in [0, 1]) is well served by
    # linear bins; turn this on for benchmark families whose metric can vary
    # over orders of magnitude (e.g. regression losses like MSE/RMSE), where
    # linear bins over/under-resolve depending on which tail dominates the
    # observed range. Applies to both the causal and oracle
    # (`metric_range_override`) ranges.
    metric_log_scale: bool = False

    def add_trial(self, config, result, advantage=None):

        trial = Trial(config, result, advantage)
        self.trials.append(trial)

    def _resolve_hypers(self, shuffle=False, hp_order: list = None):
        if hp_order is not None:
            hypers = hp_order
        else:
            hypers = list(self.config_space.items())
            if shuffle:
                random.shuffle(hypers)
            # sort hyperparameters: continuous first, categorical last
            continues_hypers = []
            categorical_hypers = []
            for hp_name, hp in hypers:
                if not isinstance(hp, Domain):
                    continue
                if isinstance(hp, Categorical):
                    categorical_hypers.append((hp_name, hp))
                else:
                    continues_hypers.append((hp_name, hp))
            hypers = continues_hypers + categorical_hypers
        return hypers

    def _build_header(self, hypers):
        string = ""
        if not self.remove_names:
            string += f"benchmark:{self.name},"
        string += f"algorithm:{self.algorithm},"
        string += f"search-space:"
        for hp_name, hp in hypers:
            string += "{"
            if not self.remove_names:
                string += f"name:{hp_name},"

            if isinstance(hp, Categorical):

                string += f"type:CAT,"
                string += f"categories:{[i for i in range(len(hp.categories))]}".replace(" ", "")
            elif isinstance(hp, Float):
                    string += f"type:UNI,"
                    string += f"min_value:{hp.lower},"
                    string += f"max_value:{hp.upper},"
                    string += f"log_scale" if is_log_space(hp) else f"linear_scale"
            elif isinstance(hp, Integer):
                    string += f"type:INT,"
                    string += f"min_value:{hp.lower},"
                    string += f"max_value:{hp.upper},"
                    string += f"log_scale" if is_log_space(hp) else f"linear_scale"
            elif isinstance(hp, FiniteRange):
                if hp.cast_int:
                    string += f"type:INT,"
                else:
                    string += f"type:UNI,"
                string += f"min_value:{hp.lower},"
                string += f"max_value:{hp.upper},"
                string += f"log_scale" if is_log_space(hp) else f"linear_scale"
            else:
                raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")
            string += "}"

        string += ',history:'
        return string

    def _build_trial_span(self, trial, hypers, y_token):
        string = ""
        for i, (hp_name, hp) in enumerate(hypers):
            if not isinstance(hp, Domain):
                continue
            if i > 0:
                string += ","

            hp_encoded = encode(trial.config[hp_name], hp, hp_name=hp_name, q=self.num_numeric_tokens)
            string += str(hp_encoded)
        string += f"*"
        string += f"{y_token}"
        string += f"|"
        return string

    def _rank_tokens(self):
        """Quantile/rank y-tokens: average 0-based rank of each trial's metric
        among all trials of this History, scaled to [0, q-1]. Ties share a
        token; the best trial is always 0 and the worst q-1."""
        ys = np.asarray([float(trial.metric) for trial in self.trials])
        if np.isnan(ys).any():
            # match quantize(): int(round(nan)) raises, so a NaN trial drops
            # the whole experiment at compile time rather than being ranked last
            raise ValueError("NaN metric in History; cannot rank")
        n = len(ys)
        if n <= 1:
            return [0] * n
        _, inverse, counts = np.unique(ys, return_inverse=True, return_counts=True)
        first_rank = np.cumsum(counts) - counts
        avg_rank = first_rank + (counts - 1) / 2
        u = avg_rank[inverse] / (n - 1)
        return np.rint(u * (self.num_numeric_tokens - 1)).astype(int).tolist()

    def _metric_tokens(self):
        """For each trial t (in order), the integer y-token to emit. In
        "quantile" mode these are rank tokens (see `_rank_tokens`); otherwise
        the metric is clipped to and linearly quantized against the per-trial
        (y_min, y_max) from `_metric_ranges`."""
        if self.metric_quantization_mode == "quantile":
            if self.metric_range_override is not None:
                raise ValueError(
                    "metric_range_override is incompatible with metric_quantization_mode='quantile' "
                    "(rank tokens do not use a range); drop the override or pick a range-based mode"
                )
            return self._rank_tokens()

        tokens = []
        for trial, (y_min, y_max) in zip(self.trials, self._metric_ranges()):
            metric_val = symlog(trial.metric) if self.metric_log_scale else trial.metric
            clipped_metric = min(max(metric_val, y_min), y_max)
            tokens.append(quantize(clipped_metric, y_min, y_max, q=self.num_numeric_tokens))
        return tokens

    def _metric_ranges(self):
        """For each trial t (in order), the (y_min, y_max) quantization range
        to use. If `metric_range_override` is set, every trial uses that
        fixed range (a debug/ablation mode -- see its docstring). Otherwise,
        dispatches on `metric_quantization_mode`:

        - "original" (default, matches upstream bbo-pile): a single GLOBAL
          range spanning every trial in this History -- computed once from
          the full trial list, so later trials influence the range used for
          earlier ones. No percentile clipping.
        - "causal": the opt-in quantization-causal-robust fix. The range for
          trial t is computed only from trials observed up to and including
          t (never from later trials, matching what is available at
          inference time). The upper (worse) tail is percentile-clipped by
          default to stop a single diverged trial from consuming the token
          resolution; the lower (better) tail is left uncapped by default
          since resolution among the best trials matters most for search.

        If `metric_log_scale` is set, ranges are computed on the symlog'd
        metric (see `symlog`) -- `_build_trial_span` symlogs each trial's
        metric the same way before clipping/quantizing against these ranges,
        so both stay in the same (transformed) space.
        """
        transform = symlog if self.metric_log_scale else (lambda x: x)

        if self.metric_range_override is not None:
            y_min, y_max = self.metric_range_override
            y_min, y_max = transform(y_min), transform(y_max)
            if y_min == y_max:
                y_max += 1  # Avoid division by zero in quantization
            return [(y_min, y_max)] * len(self.trials)

        if self.metric_quantization_mode == "original":
            observed_metrics = [transform(trial.metric) for trial in self.trials]
            y_min = min(observed_metrics)
            y_max = max(observed_metrics)
            if y_min == y_max:
                y_max += 1  # Avoid division by zero in quantization
            return [(y_min, y_max)] * len(self.trials)

        if self.metric_quantization_mode != "causal":
            raise ValueError(
                f"Unsupported metric_quantization_mode: {self.metric_quantization_mode!r} "
                "(expected 'original', 'causal' or 'quantile')"
            )

        observed_metrics = []
        ranges = []
        for trial in self.trials:
            observed_metrics.append(transform(trial.metric))
            y_min = np.percentile(observed_metrics, self.metric_percentile_lo)
            y_max = np.percentile(observed_metrics, self.metric_percentile_hi)
            if y_min == y_max:
                y_max += 1  # Avoid division by zero in quantization
            ranges.append((y_min, y_max))
        return ranges

    def get_prompt(self, shuffle=False, hp_order: list = None):
        hypers = self._resolve_hypers(shuffle=shuffle, hp_order=hp_order)
        string = self._build_header(hypers)

        if len(self.trials) > 0:
            for trial, y_token in zip(self.trials, self._metric_tokens()):
                string += self._build_trial_span(trial, hypers, y_token)
        return string

    def get_prompt_with_weights(self, advantages: list = None, shuffle=False,
                                 hp_order: list = None, base_weight: float = 1.0):
        """Same construction as get_prompt(), but also returns a list of
        (text_chunk, weight) segments spanning the returned string: the
        shared benchmark/algorithm/search-space header gets `base_weight`,
        and each trial's `{config}*{y}|` span gets `advantages[i]` (falling
        back to `trial.advantage`, then `base_weight`, if not provided).
        """
        hypers = self._resolve_hypers(shuffle=shuffle, hp_order=hp_order)
        header = self._build_header(hypers)
        segments = [(header, base_weight)]

        if len(self.trials) > 0:
            for i, (trial, y_token) in enumerate(zip(self.trials, self._metric_tokens())):
                span = self._build_trial_span(trial, hypers, y_token)
                if advantages is not None and i < len(advantages):
                    weight = advantages[i]
                elif trial.advantage is not None:
                    weight = trial.advantage
                else:
                    weight = base_weight
                segments.append((span, weight))

        string = "".join(text for text, _ in segments)
        return string, segments

    @classmethod
    def from_syne_tune_experiment(cls, experiment: ExperimentResult,
                                  num_numeric_tokens: int = 1000,
                                  remove_names: bool = False,
                                  max_num_trials: int = None,
                                  metric_quantization_mode: str = "original",
                                  metric_range_override: tuple = None,
                                  metric_log_scale: bool = False):
        """
        Create a History object from a Syne Tune ExperimentResult.

        `metric_quantization_mode`: see `History.metric_quantization_mode` --
        "original" (default) matches upstream bbo-pile's global-range
        quantization; "causal" opts into the quantization-causal-robust fix;
        "quantile" emits per-History rank tokens instead of a range.

        `metric_range_override`: see `History.metric_range_override` --
        pass a benchmark's true (y_min, y_max) here to build a History that
        quantizes against that fixed range instead of
        `metric_quantization_mode`.

        `metric_log_scale`: see `History.metric_log_scale` -- set for
        benchmark families whose metric varies over orders of magnitude
        (e.g. regression losses), rather than a fixed bounded scale.
        """
        metadata = experiment.metadata
        config_space = config_space_from_json_dict(json.loads(metadata['config_space']))
        metric_name = metadata["metric_names"][0]
        results = experiment.results
        mode = metadata['metric_mode'] if 'metric_mode' in metadata else 'min'
        benchmark_name = metadata['benchmark'] if 'benchmark' in metadata else metadata['entrypoint']
        algorithm_name = metadata['algorithm'] if 'algorithm' in metadata else metadata['scheduler_name']
        hist = cls(config_space=config_space,
                        name=benchmark_name,
                        algorithm=algorithm_name,
                        metric_names=metric_name,
                   num_numeric_tokens=num_numeric_tokens,
                   remove_names=remove_names,
                   metric_quantization_mode=metric_quantization_mode,
                   metric_range_override=metric_range_override,
                   metric_log_scale=metric_log_scale)

        for i, (trial_id, trial) in enumerate(results.groupby('trial_id')):
            row = trial.iloc[-1]
            config = {k: row[f"config_{k}"] for k in config_space.keys()}
            result = row[metric_name]
            if mode == 'max':
                result = -result
            hist.add_trial(config, result)
            if max_num_trials is not None and i >= max_num_trials - 1:
                break

        return hist

if __name__ == "__main__":
    from syne_tune.config_space import uniform, randint, choice, loguniform
    config_space = {
        'x': uniform(0, 1),
        'y': randint(0, 10),
        'z': choice(['a', 'b', 'c']),
        'log_x': loguniform(0.01, 1),
    }
    history = History(name='test', algorithm='test', config_space=config_space)
    history.add_trial({'x': 0.5, 'y': 5, 'z': 'a', 'log_x': 0.1}, 0.5)
    history.add_trial({'x': 0.6, 'y': 6, 'z': 'b', 'log_x': 0.9}, 0.6)
    prompt = history.get_prompt()
    print(prompt)

    rank_history = History(name='test', algorithm='test', config_space=config_space,
                           metric_quantization_mode="quantile")
    cfg = {'x': 0.5, 'y': 5, 'z': 'a', 'log_x': 0.1}
    for metric in [0.3, 0.1, 0.3, 1e9, 0.2]:
        rank_history.add_trial(cfg, metric)
    # ranks 0..4 with the two 0.3s sharing average rank 2.5 -> 2.5/4*999
    assert rank_history._metric_tokens() == [624, 0, 624, 999, 250], rank_history._metric_tokens()
    print(rank_history.get_prompt())

    test_cases = [
        (0.1, 0.01, 1, True),
        (0.99, 0.01, 1, True),
        (0.2, 0.0, 1.0, False),
        (100, 0, 200, False),
        (0.01, 0.01, 1, True),
        (50, 10, 100, True),
        (0.5, 0.0, 1.0, False),
        (1, 1, 10, True),
        (5, 1, 10, False),
    ]
    q = 1000
    for x, x_min, x_max, log_scale in test_cases:
        x_quantized = quantize(x, x_min, x_max, q, log_scale)
        x_dequantized = dequantize(x_quantized, x_min, x_max, q, log_scale)

        if log_scale:
            expected_error = np.exp(np.log(x_max + 1e-10) - np.log(x_min + 1e-10) / (2 * (q - 1)))
        else:
            expected_error = (x_max - x_min) / (2 * (q - 1))

        assert abs(x - x_dequantized) < expected_error, (
            f"x: {x}, x_min: {x_min}, x_max: {x_max}, log_scale: {log_scale}, "
            f"x_quantized: {x_quantized}, x_dequantized: {x_dequantized}"
        )
