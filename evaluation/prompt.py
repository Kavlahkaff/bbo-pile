"""The one place that defines the prompt, used by the export and by the rollouts."""
from syne_tune.config_space import uniform

from open_optformer.history import History

CONFIG_SPACE = {"x0": uniform(0.0, 1.0), "x1": uniform(0.0, 1.0)}
TOKENS_PER_TRIAL = 6  # x0 , x1 * y |


def make_history(tag, k=(), y=(), n_init=None):
    """k: points on the token grid (0..999), y: values. n_init: size of the initial design, to mark its end; None = no marker."""
    h = History(name="f", algorithm=tag, config_space=dict(CONFIG_SPACE), remove_names=True, space_first=True, n_init=n_init)
    for (k0, k1), v in zip(k, y):
        h.add_trial({"x0": k0 / 999, "x1": k1 / 999}, float(v))
    return h
