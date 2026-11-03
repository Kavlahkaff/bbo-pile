"""
Shared helpers for the GP-BO trajectory corpus (generator: master-thesisv2/gp_traj.py, preset "student").

Corpus layout (one directory per corpus, several directories may be combined; fids are unique across them):
    traj_NNN.jsonl       one trajectory per line, duplicate keys possible (keep the last)
    runs_NNN.jsonl       one summary per run
    functions_NNN.json   list of function specs
    functions/<fid>.npy  512x512 float32 grids (test functions only)

Conventions verified against gp_traj.Fn: X[:, 0] indexes grid rows, so spec["argmin"][0] is the x0 coordinate.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from open_optformer.history import quantize

KINDS = ["rbf", "m12", "m32", "m52"]
LS_RANGE = (0.05, 0.5)  # generator ranges (log-uniform), used for the hp suffix
OS_RANGE = (0.25, 4.0)
FLOOR = 1e-3  # regret floor matching the 1000-level quantization
EPS = 1e-8  # as in gp_traj: log_regret = log10(running min yn + EPS)


# ---------------------------------------------------------------------------- reproducible randomness
def seed_of(*parts) -> int:
    """Same hashing scheme as gp_traj.seed_of: reproducible, independent of process / file order."""
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % 2**62


def rng_of(*parts) -> np.random.Generator:
    return np.random.default_rng(seed_of(*parts))


# ---------------------------------------------------------------------------- teacher ids
def parse_mid(mid: str) -> dict:
    """'true|ucb4|rbf' -> {mode: 'true', acq: 'ucb4', kernel: 'rbf'}; 'rs' -> {mode: 'rs', ...}."""
    if mid == "rs":
        return {"mode": "rs", "acq": None, "kernel": None}
    mode, acq, kernel = mid.split("|")
    return {"mode": mode, "acq": acq, "kernel": kernel}


def is_oracle(mid: str) -> bool:
    return parse_mid(mid)["mode"] == "true"


def is_mle(mid: str) -> bool:
    return parse_mid(mid)["mode"] == "mle"


def is_matched(mid: str, fkind: str) -> bool:
    return parse_mid(mid)["kernel"] == fkind


# ---------------------------------------------------------------------------- the 8 square symmetries
# x' = A (x - 1/2) + 1/2 with A a signed 2x2 permutation matrix. Index 0 is the identity.
SYMMETRIES = [
    np.array([[1, 0], [0, 1]]),    # 0 identity
    np.array([[-1, 0], [0, 1]]),   # 1 flip x0
    np.array([[1, 0], [0, -1]]),   # 2 flip x1
    np.array([[-1, 0], [0, -1]]),  # 3 rotation 180
    np.array([[0, 1], [1, 0]]),    # 4 transpose (swap axes)
    np.array([[0, -1], [1, 0]]),   # 5 rotation 90
    np.array([[0, 1], [-1, 0]]),   # 6 rotation 270
    np.array([[0, -1], [-1, 0]]),  # 7 anti-transpose
]
SYMMETRY_SETS = {
    "d4": list(range(8)),          # all 8
    "noswap": [0, 1, 2, 3],        # no axis swap: ARD lengthscales unchanged
    "rot": [0, 5, 3, 6],           # rotations (90/270 DO swap the axes)
}


def apply_sym(X, g: int) -> np.ndarray:
    """Map points X (n x 2, in [0,1]^2) through symmetry g."""
    X = np.asarray(X, dtype=float)
    return (X - 0.5) @ SYMMETRIES[g].T + 0.5


def inverse(g: int) -> int:
    At = SYMMETRIES[g].T
    return next(h for h, B in enumerate(SYMMETRIES) if np.array_equal(B, At))


def compose(g: int, h: int) -> int:
    """Index of the symmetry x -> g(h(x))."""
    C = SYMMETRIES[g] @ SYMMETRIES[h]
    return next(k for k, B in enumerate(SYMMETRIES) if np.array_equal(B, C))


def swaps_axes(g: int) -> bool:
    return SYMMETRIES[g][0, 0] == 0


def transform_grid(grid: np.ndarray, g: int) -> np.ndarray:
    """Grid of f_g(x) = f(g^-1 x) on the same nodes i/(n-1). Exact, since the nodes are symmetric."""
    n = grid.shape[0]
    idx = np.arange(n)
    I, J = np.meshgrid(idx, idx, indexing="ij")
    P = np.stack([I.ravel(), J.ravel()], 1) / (n - 1)          # node coordinates of the new grid
    Q = np.rint(apply_sym(P, inverse(g)) * (n - 1)).astype(int)  # g^-1 of each node, as indices
    return grid[Q[:, 0], Q[:, 1]].reshape(n, n)


def sample_symmetries(k: int, key: str, seed: int = 0, sym_set: str = "d4") -> list[int]:
    """k distinct symmetries from sym_set, identity always first; reproducible per trajectory key."""
    pool = SYMMETRY_SETS[sym_set]
    assert 1 <= k <= len(pool), f"--symmetries must be in 1..{len(pool)} for set {sym_set}"
    others = [g for g in pool if g != 0]
    rest = rng_of("sym", seed, key).permutation(others)[: k - 1].tolist() if k > 1 else []
    return [0] + [int(g) for g in rest]


# ---------------------------------------------------------------------------- function evaluation
def bilinear(grid: np.ndarray, X) -> np.ndarray:
    """Bilinear interpolation of an n x n grid on [0,1]^2, identical to gp_traj.Fn.__call__."""
    n = grid.shape[0]
    p = np.clip(np.atleast_2d(np.asarray(X, float)), 0, 1) * (n - 1)
    i = np.minimum(p.astype(int), n - 2)
    t, g, a, b = p - i, grid, i[:, 0], i[:, 1]
    return ((1 - t[:, 0]) * (1 - t[:, 1]) * g[a, b] + t[:, 0] * (1 - t[:, 1]) * g[a + 1, b]
            + (1 - t[:, 0]) * t[:, 1] * g[a, b + 1] + t[:, 0] * t[:, 1] * g[a + 1, b + 1])


# ---------------------------------------------------------------------------- function geometry
def edge_distance(argmin) -> float:
    a0, a1 = argmin
    return float(min(a0, a1, 1 - a0, 1 - a1))


def near_boundary(spec: dict, margin: float) -> bool:
    return edge_distance(spec["argmin"]) <= margin


def quantize_hp(spec: dict, g: int = 0, q: int = 1000) -> tuple[int, int, int]:
    """Log-quantized true hyperparameters (ls0, ls1, os) of the function as seen after symmetry g."""
    ls = list(spec["ls"])
    if swaps_axes(g):
        ls = ls[::-1]
    qs = [quantize(l, *LS_RANGE, q=q, log_scale=True) for l in ls]
    qo = quantize(spec["var"], *OS_RANGE, q=q, log_scale=True)
    clip = lambda v: int(min(max(v, 0), q - 1))
    return clip(qs[0]), clip(qs[1]), clip(qo)


# ---------------------------------------------------------------------------- regret metrics
def log_regret_from_y(y, fmin: float, fmax: float) -> np.ndarray:
    yn = (np.asarray(y, float) - fmin) / (fmax - fmin)
    return np.log10(np.minimum.accumulate(yn) + EPS)


def auc(lr, floor: float | None = None) -> float:
    lr = np.asarray(lr, float)
    if floor is not None:
        lr = np.maximum(lr, math.log10(floor))
    return float(lr.mean())


# ---------------------------------------------------------------------------- corpus reading
def traj_files(corpus_dirs: Iterable) -> list[Path]:
    files = []
    for d in corpus_dirs:
        files += sorted(Path(d).glob("traj_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no traj_*.jsonl in {list(corpus_dirs)}")
    return files


def load_specs(corpus_dirs: Iterable) -> dict[str, dict]:
    """fid -> spec over all corpus directories. A fid appearing in two directories is an error."""
    specs, origin = {}, {}
    for d in corpus_dirs:
        for f in sorted(Path(d).glob("functions_*.json")):
            for s in json.load(open(f)):
                if s["fid"] in origin and origin[s["fid"]] != str(d):
                    raise ValueError(f"fid {s['fid']} appears in {origin[s['fid']]} and {d}")
                origin[s["fid"]] = str(d)
                specs[s["fid"]] = s
    return specs


def read_traj_file(path, fields: tuple | None = None) -> dict[str, dict]:
    """key -> trajectory (last line wins). Truncated lines (crash) are skipped. If `fields` is given, only those
    are kept, which saves memory when the per-step diagnostics are not needed."""
    out = {}
    with open(path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if fields is not None:
                d = {k: d[k] for k in fields if k in d}
            out[d["key"]] = d
    return out


def iter_trajectories(corpus_dirs: Iterable, fields: tuple | None = None) -> Iterator[dict]:
    for f in traj_files(corpus_dirs):
        yield from read_traj_file(f, fields).values()


def git_commit(path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
