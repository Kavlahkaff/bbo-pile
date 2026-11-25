from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.prompt import make_history
from generate_training_data.compile_traces import lines_of
from open_optformer.history import MARKER, log_tokens


def run(n=20):
    r = np.random.default_rng(1)
    return r.integers(0, 1000, (n, 2)).tolist(), (10.0 ** -r.uniform(0, 7, n)).tolist()


def test_log():
    t = log_tokens([1.0, 0.0, 1e-4])
    assert t[0] == 999 and t[1] == 0 and 0 < t[2] < 999
    assert log_tokens([2.0, 2.0]) == [0, 0]


def test_no_marker_by_default():
    k, y = run()
    assert MARKER not in make_history("GP", k, y).get_prompt()


@pytest.mark.parametrize("n_init", range(5, 11))
def test_marker_position(n_init):
    k, y = run()
    for m in range(len(k) + 1):
        plain, marked = make_history("NM", k[:m], y[:m]).get_prompt(), make_history("NM", k[:m], y[:m], n_init).get_prompt()
        assert marked.replace(MARKER, "") == plain
        assert marked.count(MARKER) == (m >= n_init)
        if m >= n_init:
            assert marked.split(MARKER)[0].count("|") == n_init and marked.split(MARKER)[0].endswith("|")


@pytest.mark.parametrize("marker", [False, True])
def test_export_equals_inference_prompt(marker):
    k, y = run()
    t = {"opt": "NM", "family": "rosen", "k": k, "y": y, "n_init": 7}
    L = lines_of(t, SimpleNamespace(marker=marker))
    assert L == [make_history(g, k, y, 7 if marker else None).get_prompt() for g in ("NM", "BEST")]
    assert "algorithm:BEST,history:" in L[1]
    assert len(lines_of({**t, "opt": "GP"}, SimpleNamespace(marker=marker))) == 1


def test_student_prompt():
    rollout = pytest.importorskip("evaluation.rollout")
    k, y = run()
    s = SimpleNamespace(marker=True, tok=SimpleNamespace(encode=lambda p, bos: np.array([p])))
    assert rollout.Student.prompt(s, "GP", k, y, 6) == [make_history("GP", k, y, 6).get_prompt()]
    s.marker = False
    assert rollout.Student.prompt(s, "GP", k, y, 6) == [make_history("GP", k, y).get_prompt()]
