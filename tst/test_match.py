import json

import numpy as np
import pytest

from datagen.functions import FAMILIES, Function
from datagen.optimizers import OPTIMIZERS, initial_design
from evaluation.match import (Distances, function_rows, features, grid, histories, main, mean_dist, nm_next, teacher_function,
                              next_distribution, near, teacher_run)

M = 6


@pytest.fixture(scope="module")
def teacher():
    return json.loads(teacher_function(("rosen", 100_000, 0, M)))


def test_distance_is_unbiased_and_positive():
    r = np.random.default_rng(0)
    same, far = [], []
    for _ in range(300):
        A, B = r.normal(size=(2, 5, 3, 2))
        at, upto = Distances({"a": A, "b": B, "c": B + 2})("a", "b")
        same.append([at[1], upto[2]])
        far.append(Distances({"a": A, "c": B + 2})("a", "c")[0][1])
    same = np.array(same)
    assert (np.abs(same.mean(0)) < 4 * same.std(0) / np.sqrt(len(same))).all()
    assert np.mean(far) > 1


def test_mean_dist_excludes_own_run():
    A = np.random.default_rng(1).normal(size=(4, 3, 2))
    D = mean_dist(A, A, True)
    want = np.mean([np.linalg.norm(A[i, 0] - A[j, 2]) for i in range(4) for j in range(4) if i != j])
    assert np.isclose(D[0, 2], want)


@pytest.mark.parametrize("family", FAMILIES)
def test_nm_next_keeps_the_history(family):
    fn = Function(family, 100_003)
    X = teacher_run("NM", fn, 0, 0)
    n0 = len(initial_design(fn.i, 0))
    for n in range(n0, 50):                                      # asserts inside that the replay reproduces X[:n]
        for d in range(4):
            nm_next(fn, 0, 0, X, n, d)
    a = {tuple(grid(nm_next(fn, 0, 0, X, n0 + 1, d))) for d in range(20)}
    assert tuple(grid(X[n0 + 1])) in a and len(a) <= 2           # first edge of the first simplex: one unknown sign


def oracle(d, tag, opt):
    """A 'student' that is the teacher `opt` with other random numbers."""
    fn = Function(d["family"], d["i"])
    X = [teacher_run(opt, fn, d["seed"], 1000 + m) for m in range(M)]
    nxt = {f"{h}|{k}": d["next"][f"{h}|{k}"].get(opt, d["next"][f"{h}|{k}"]["GP"]) for h, k in histories()}
    return {(tag, "run"): {"k": [grid(x) for x in X], "y": [fn(x).tolist() for x in X]}, (tag, "next"): {"next": nxt, "mass": {f"{h}|{k}": {} for h, k in histories()}}}


def test_oracle_is_nearest_to_its_teacher(teacher):
    for opt in OPTIMIZERS:
        rows = {r[:4]: r[4] for r in function_rows((teacher, oracle(teacher, "BEST", opt)))}
        c = {o: rows["run-upto", 40, "xy", f"S:BEST>{o}"] for o in OPTIMIZERS}
        assert min(c, key=c.get) == opt
        assert abs(c[opt]) < 0.1 * max(c.values())


def test_features_step(teacher):
    f = features(teacher["k"]["NM"], teacher["y"]["NM"], teacher["n_init"])
    assert f["xy"].shape == (M, 40, 2) and f["step"].shape == (M, 40, 1)
    assert np.allclose(f["step"][:, 0], -3)                      # NM evaluates the incumbent again first


def test_end_to_end_report(tmp_path):
    T = tmp_path / "teachers.jsonl"
    main(["teachers", "--functions", "2", "--reps", str(M), "--workers", "2", "--output", str(T)])
    text = T.read_text()
    T.write_text(text[:-200])                                    # cut line: that function is computed again
    main(["teachers", "--functions", "2", "--reps", str(M), "--workers", "2", "--output", str(T)])
    D = []
    for l in T.read_text().splitlines():
        try:
            D.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    assert len({d["fid"] for d in D}) == len(D) == 2 * len(FAMILIES)
    with open(tmp_path / "run_0.jsonl", "w") as fh:
        for d in D:
            for (tag, what), v in oracle(d, "BEST", "NM").items():
                fh.write(json.dumps({"fid": d["fid"], "tag": tag, "what": what, **v}) + "\n")
    out = tmp_path / "report.txt"
    main(["report", "--teachers", str(T), "--run", f"o={tmp_path}/run", "--run", f"o={tmp_path}/run", "--workers", "2", "--output", str(out)])
    t = out.read_text()
    assert "tag BEST, run-upto, space xy, distance, 2 model seed(s)" in t and "MISSING" not in t and "WARNING" not in t
    assert (tmp_path / "report.csv.gz").exists()


def test_shift_and_spread():
    r = np.random.default_rng(0)
    sh, sp = [], []
    for _ in range(400):
        D = Distances({"S:a": r.normal(size=(6, 1, 2)), "T:b": r.normal(size=(5, 1, 2)) * 3 + [2, 0]})
        sh.append(D.shift2("S:a", "T:b")[0][0])
        sp.append([D.spread("S:a")[0][0], D.spread("T:b")[0][0]])
    assert abs(np.mean(sh) - 4) < 4 * np.std(sh) / 20
    assert np.allclose(np.mean(sp, 0) / np.sqrt(np.pi), [1, 3], rtol=0.06)      # E|A-A'| = sigma sqrt(pi) in 2D


class FakeStudent:
    """p(k0) and p(k1 | k0) from a known joint distribution; the prompt only carries k0."""
    y_norm, comma, numbers = "causal_log", -1, list(range(1000))

    class tok:
        encode = staticmethod(lambda p, bos: np.array([0]))

    def __init__(self, J):
        import torch
        self.p0, self.p1 = torch.tensor(J.sum(1)).float(), torch.tensor(J / J.sum(1, keepdims=True)).float()

    def number_probs(self, ids, batch):
        import torch
        return torch.stack([self.p0 if len(s) == 1 else self.p1[s[1]] for s in ids])


def test_mass_is_exact(teacher):
    pytest.importorskip("torch")
    a, b = teacher["next"]["init|1"]["GP"][0], teacher["next"]["init|1"]["NM"][0]
    J = np.full((1000, 1000), 0.1 / 1e6)
    J[a[0], a[1]] += 0.6
    J[b[0], b[1]] += 0.3
    out = next_distribution(FakeStudent(J), teacher, "BEST", 200, 64)
    m = out["mass"]["init|1"]
    for o in ("GP", "NM"):
        assert np.isclose(m[f"{o}|2"], J[near(teacher["next"]["init|1"][o], 2)].sum(), atol=1e-4)
    share = np.mean([p == a for p in out["next"]["init|1"]])
    assert abs(share - 0.6) < 0.15 and len(out["next"]["init|1"]) == 200
