import numpy as np
import pytest

from open_optformer.gp_corpus import (SYMMETRIES, SYMMETRY_SETS, apply_sym, bilinear, compose, edge_distance,
                                      inverse, quantize_hp, sample_symmetries, swaps_axes, transform_grid)


def test_symmetries_form_a_group():
    assert np.array_equal(SYMMETRIES[0], np.eye(2))
    for g in range(8):
        assert compose(g, inverse(g)) == 0
        for h in range(8):
            compose(g, h)  # closed: raises StopIteration otherwise


def test_symmetries_map_unit_square_to_itself():
    X = np.random.default_rng(0).uniform(size=(100, 2))
    for g in range(8):
        Y = apply_sym(X, g)
        assert Y.min() >= 0 and Y.max() <= 1
        assert np.allclose(apply_sym(Y, inverse(g)), X)


def test_swaps_axes_and_sets():
    assert [swaps_axes(g) for g in range(8)] == [False] * 4 + [True] * 4
    assert not any(swaps_axes(g) for g in SYMMETRY_SETS["noswap"])
    # 90/270 degree rotations swap the axes
    assert swaps_axes(5) and swaps_axes(6)
    assert np.allclose(apply_sym([[0.2, 0.7]], 5), [[0.3, 0.2]])  # (x0, x1) -> (1 - x1, x0)


def test_transform_grid_is_exact_under_bilinear():
    r = np.random.default_rng(1)
    grid = r.standard_normal((33, 33))
    X = r.uniform(size=(500, 2))
    for g in range(8):
        Gg = transform_grid(grid, g)
        # f_g(g x) == f(x): a trajectory mapped through g is a valid trajectory on f_g with unchanged y
        assert np.allclose(bilinear(Gg, apply_sym(X, g)), bilinear(grid, X), atol=1e-12)
        # argmin moves with g
        i, j = np.unravel_index(grid.argmin(), grid.shape)
        ig, jg = np.unravel_index(Gg.argmin(), Gg.shape)
        assert np.allclose(apply_sym([[i / 32, j / 32]], g), [[ig / 32, jg / 32]])


def test_sample_symmetries():
    for k in range(1, 9):
        s = sample_symmetries(k, "rbf-0001|0|rs")
        assert s[0] == 0 and len(set(s)) == k
    assert sample_symmetries(4, "a") == sample_symmetries(4, "a")
    assert set(sample_symmetries(4, "a", sym_set="noswap")) == {0, 1, 2, 3}
    with pytest.raises(AssertionError):
        sample_symmetries(5, "a", sym_set="noswap")


def test_quantize_hp_swaps_lengthscales():
    spec = {"ls": [0.05, 0.5], "var": 1.0}
    l0, l1, o = quantize_hp(spec, 0)
    assert (l0, l1) == (0, 999)
    for g in range(8):
        q = quantize_hp(spec, g)
        assert q[:2] == ((999, 0) if swaps_axes(g) else (0, 999))
        assert q[2] == o


def test_edge_distance():
    assert edge_distance([0.5, 0.5]) == 0.5
    assert edge_distance([0.01, 0.6]) == pytest.approx(0.01)
    assert edge_distance([0.3, 0.98]) == pytest.approx(0.02)
