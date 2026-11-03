import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "gp_functions"))
from gp_function import GPGridFunction  # noqa: E402
from open_optformer.gp_corpus import apply_sym, bilinear  # noqa: E402


def test_identity_matches_bilinear_and_regret():
    r = np.random.default_rng(0)
    grid = r.standard_normal((65, 65))
    f = GPGridFunction(grid, {"fid": "x", "fmin": grid.min(), "fmax": grid.max()})
    X = r.uniform(size=(50, 2))
    assert np.allclose(f.evaluate(X), bilinear(grid, X))
    assert f({"x0": float(X[0, 0]), "x1": float(X[0, 1])})["y"] == f.evaluate(X[:1])[0]
    lr = f.log_regret(f.evaluate(X))
    assert np.all(np.diff(lr) <= 0) and lr[-1] >= np.log10(1e-8) - 1e-9


def test_symmetry_reproduces_teacher_values_on_transformed_design():
    r = np.random.default_rng(1)
    grid = r.standard_normal((65, 65))
    X = r.uniform(size=(30, 2))
    y = bilinear(grid, X)
    for g in range(8):
        fg = GPGridFunction(grid, None, symmetry=g)
        assert np.allclose(fg.evaluate(apply_sym(X, g)), y)
        # minimum value is invariant
        assert np.isclose(fg.fmin, grid.min()) and np.isclose(fg.fmax, grid.max())
