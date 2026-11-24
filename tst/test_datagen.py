import json

import numpy as np
import pytest

from datagen.functions import Function
from datagen.generate import BASE, main, repair
from datagen.optimizers import OPTIMIZERS, TRIALS, initial_design, run


@pytest.mark.parametrize("family", ["gp", "rosen"])
@pytest.mark.parametrize("opt", list(OPTIMIZERS))
def test_runs(family, opt):
    f = Function(family, 3)
    X = run(opt, f, 0)
    X0 = initial_design(3, 0)
    assert X.shape == (TRIALS, 2) and (X[:len(X0)] == X0).all()
    assert np.allclose(X * 999, np.rint(X * 999)) and X.min() >= 0 and X.max() <= 1
    assert (X == run(opt, Function(family, 3), 0)).all()
    assert np.isfinite(f.log_regret(f(X))).all()


def test_rosenbrock_minimum():
    f = Function("rosen", 7)
    assert f(f.spec["c"])[0] < 1e-20


def test_generate_and_resume(tmp_path):
    args = ["--out", str(tmp_path), "--train", "0", "--limit", "2", "--workers", "1"]
    main(args)
    p = tmp_path / "traj_00000.jsonl"
    text = p.read_text()
    L = [json.loads(l) for l in text.splitlines()]
    assert len(L) == 2 * 5 * 3 and {d["i"] for d in L} == {BASE} and {d["split"] for d in L} == {"test"}
    d = L[0]
    assert np.allclose(Function(d["family"], d["i"])(np.array(d["k"]) / 999), d["y"], atol=0)
    p.write_text(text[: len(text) * 2 // 3])                          # crash in the middle of a line
    main(args)
    assert sorted(p.read_text().splitlines()) == sorted(text.splitlines())
    assert len(repair(p)) == 2
