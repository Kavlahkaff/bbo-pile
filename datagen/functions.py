"""Objective functions on [0,1]^2 (minimised). Function i of a family depends only on (family, i).

gp     sample of a GP (kernel KINDS[i % 4], log-uniform ARD lengthscales and output scale) on a 512 x 512 grid, bilinear
rosen  Rosenbrock, randomly rotated, scaled (s in [2, 8]) and shifted, b in {10, 100}; exact, f* = 0
Regret is normalised by (lo, hi): the grid range (gp), 0 and the median of f over the domain (rosen)."""
import hashlib
import math

import numpy as np

KINDS = ["rbf", "m12", "m32", "m52"]
NU = {"m12": 0.5, "m32": 1.5, "m52": 2.5}
FAMILIES = ["gp", "rosen"]
LS_RANGE, OS_RANGE, GRID = (0.05, 0.5), (0.25, 4.0), 512
EPS = 1e-8  # log10(regret + EPS)


def seed_of(*parts) -> int:
    """Reproducible seed from labels, independent of process and order."""
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % 2**62


# --------------------------------------------------------------------------------------------- GP samples
def kappa(r, kind):
    """Correlation at scaled distance r = |dx / l| (GPyTorch conventions)."""
    if kind == "rbf":
        return np.exp(-0.5 * r * r)
    s = math.sqrt(2 * NU[kind]) * r
    return {"m12": np.exp(-s), "m32": (1 + s) * np.exp(-s), "m52": (1 + s + s * s / 3) * np.exp(-s)}[kind]


def sample_gp(kind, ls, var, seed, n=GRID):
    """GP sample on an n x n grid by circulant embedding on a padded torus."""
    import scipy.fft as F
    m = n * (4 if max(ls) <= 0.3 else 6 if max(ls) <= 0.6 else 8)
    d = np.minimum(np.arange(m), m - np.arange(m)) / (n - 1)
    lam = F.fft2(kappa(np.sqrt((d[:, None] / ls[0]) ** 2 + (d[None, :] / ls[1]) ** 2), kind)).real
    z = np.random.default_rng(seed).standard_normal((2, m, m))
    f = F.fft2(np.sqrt(np.clip(lam, 0, None) / m**2) * (z[0] + 1j * z[1])).real[:n, :n]
    return f * math.sqrt(var)


def gp_spec(i):
    kind = KINDS[i % 4]
    r = np.random.default_rng(seed_of("func", kind, 10_000 + i))
    return {"kind": kind, "ls": np.exp(r.uniform(*np.log(LS_RANGE), 2)).tolist(),
            "var": float(np.exp(r.uniform(*np.log(OS_RANGE)))), "seed": int(r.integers(2**62))}


def rosen_spec(i):
    r = np.random.default_rng(seed_of("rosen", i))
    return {"kind": "rosen", "rot": float(r.uniform(0, 2 * np.pi)), "scale": float(np.exp(r.uniform(np.log(2.0), np.log(8.0)))),
            "c": r.uniform(0.2, 0.8, 2).tolist(), "b": float(r.choice([10.0, 100.0]))}


# --------------------------------------------------------------------------------------------- function object
class Function:
    def __init__(self, family, i):
        self.family, self.i, self.fid = family, int(i), f"{family}-{i}"
        if family == "gp":
            self.spec = gp_spec(i)
            g = sample_gp(self.spec["kind"], self.spec["ls"], self.spec["var"], self.spec["seed"])
            self.g = g.astype(np.float32).astype(float)
            self.lo, self.hi = float(self.g.min()), float(self.g.max())
        elif family == "rosen":
            self.spec = rosen_spec(i)
            self.lo = 0.0
            self.hi = float(np.median(self(np.random.default_rng(seed_of("rosen-med", i)).uniform(size=(4096, 2)))))
        else:
            raise ValueError(family)

    @property
    def group(self):
        """Subgroup used in the reports: kernel, or Rosenbrock b."""
        return self.spec["kind"] if self.family == "gp" else f"b{int(self.spec['b'])}"

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if self.family == "rosen":
            s = self.spec
            c, sn = np.cos(s["rot"]), np.sin(s["rot"])
            U = 1.0 + s["scale"] * (X - np.asarray(s["c"])) @ np.array([[c, -sn], [sn, c]]).T
            return (1 - U[:, 0]) ** 2 + s["b"] * (U[:, 1] - U[:, 0] ** 2) ** 2
        n, g = GRID, self.g
        p = np.clip(X, 0, 1) * (n - 1)
        i = np.minimum(p.astype(int), n - 2)
        t, a, b = p - i, i[:, 0], i[:, 1]
        return ((1 - t[:, 0]) * (1 - t[:, 1]) * g[a, b] + t[:, 0] * (1 - t[:, 1]) * g[a + 1, b]
                + (1 - t[:, 0]) * t[:, 1] * g[a, b + 1] + t[:, 0] * t[:, 1] * g[a + 1, b + 1])

    def log_regret(self, y):
        """log10 of the best normalised regret so far (+ EPS), one value per trial."""
        return np.log10(np.minimum.accumulate((np.asarray(y, float) - self.lo) / (self.hi - self.lo)) + EPS)
