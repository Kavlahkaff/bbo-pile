import numpy as np
import pytest

from evaluation.prompt import make_history
from open_optformer.history import causal_log_tokens, log_tokens


def ys(seed, n=50):
    r = np.random.default_rng(seed)
    return (10.0 ** -r.uniform(0, 7, n)).tolist()


@pytest.mark.parametrize("seed", range(20))
def test_causal_log(seed):
    y = ys(seed)
    t = causal_log_tokens(y)
    assert min(t) >= 0 and max(t) <= 999 and t[0] == 499
    for n in range(1, len(y)):                                       # append-only
        assert causal_log_tokens(y[:n]) == t[:n]
        assert (t[n] < 499) == (y[n] < min(y[:n]))                   # every improvement is visible
    z = (np.array(y) * 4.0).tolist()
    assert causal_log_tokens(z) == t                                 # scale invariant


def test_causal_log_edges():
    assert causal_log_tokens([]) == []
    assert causal_log_tokens([1.0, 1.0, 1.0]) == [499, 499, 499]
    assert causal_log_tokens([1.0, 0.0]) == [499, 0]
    assert causal_log_tokens([1.0, 2.0]) == [499, 999]
    t = causal_log_tokens([1.0, 0.0, 1e-3, 0.0, 1.0, -1e-12])
    assert t[3] == 499 and 499 < t[2] < t[4] == 999 and t[5] == 498


def test_log():
    t = log_tokens([1.0, 0.0, 1e-4])
    assert t[0] == 999 and t[1] == 0 and 0 < t[2] < 999
    assert log_tokens([2.0, 2.0]) == [0, 0]


def test_prompt_prefix_property():
    k, y = [[j, 999 - j] for j in range(20)], ys(1, 20)
    full = make_history("GP", "causal_log", k, y).get_prompt()
    assert full.startswith(make_history("GP", "causal_log", k[:7], y[:7]).get_prompt())
    assert ",algorithm:GP,history:0,999*499|1,998*" in full
