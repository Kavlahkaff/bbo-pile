"""The optimizers (teachers). All minimise f on the 1000-level token grid: the shared initial design and every proposal
are k / 999, so a model that emits number tokens can express every point. 50 trials including the initial design."""
import numpy as np
from scipy.optimize import minimize

from datagen import gp_np
from datagen.functions import seed_of

TRIALS, N_INIT, Q = 50, (5, 10), 1000
OPTIMIZERS = {"GP": "m32", "NM": "nm", "RS": "rs"}                # tag -> label used in the random seeds
WINNER = {"gp": "GP", "rosen": "NM"}                               # best optimizer per family (confirmed on 1000 functions)
BETA, LS_BOUNDS, OS_BOUNDS, RESTARTS, RAW = 4.0, (0.02, 2.0), (0.05, 20.0), 8, 256


def snap(x):
    return np.round(np.clip(x, 0, 1) * (Q - 1)) / (Q - 1)


def initial_design(i, seed):
    """Uniform points, 5-10 of them, shared by all optimizers of (function, seed)."""
    r = np.random.default_rng(seed_of("init", i, seed))
    return snap(r.uniform(size=(int(r.integers(N_INIT[0], N_INIT[1] + 1)), 2)))


def random_search(key, f, X):
    r = np.random.default_rng(seed_of("rs", key))
    return snap(np.vstack([X, r.uniform(size=(TRIALS - len(X), 2))]))


def gp_ucb(key, f, X):
    """GP-UCB (Matern-3/2, beta 4), hyperparameters refitted by maximum likelihood at every step."""
    fallback = np.random.default_rng(seed_of("fallback", key))
    while len(X) < TRIALS:
        z = -f(X)
        z = (z - z.mean()) / max(z.std(ddof=1), 1e-9)
        try:
            ls, var, _ = gp_np.fit_mle(X, z, "m32", LS_BOUNDS, OS_BOUNDS, noise=1e-6)
            x, _ = gp_np.propose_ucb(gp_np.GP(X, z, "m32", ls, var, 1e-6), BETA, RESTARTS, RAW,
                                     np.random.default_rng(seed_of(key, len(X))))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            x = fallback.uniform(size=2)
        X = np.vstack([X, snap(x)])
    return X


class _Budget(Exception):
    pass


def nelder_mead(key, f, X, size=0.1):
    """Nelder-Mead from the best initial point (simplex edge `size`); on convergence restarted from the incumbent with
    half the size (at least 1e-3) until the budget is spent."""
    r = np.random.default_rng(seed_of("nm", key.rsplit("|", 1)[0]))
    pts, ys = list(X), list(f(X))

    def obj(x):
        if len(pts) >= TRIALS:
            raise _Budget
        x = snap(x)
        v = float(f(x[None])[0])
        pts.append(x)
        ys.append(v)
        return v

    while len(pts) < TRIALS:
        x0 = pts[int(np.argmin(ys))]
        sgn = r.choice([-1.0, 1.0], 2)
        sgn = np.where(x0 + sgn * size > 1, -1.0, np.where(x0 + sgn * size < 0, 1.0, sgn))   # stay inside [0,1]^2
        try:
            minimize(obj, x0, method="Nelder-Mead", bounds=[(0, 1), (0, 1)],
                     options={"initial_simplex": np.vstack([x0, x0 + np.diag(sgn * size)]), "xatol": 1e-4,
                              "fatol": 1e-10, "maxfev": TRIALS})
        except _Budget:
            break
        size = max(size / 2, 1e-3)
    return np.vstack(pts)[:TRIALS]


def run(tag, fn, seed):
    """Points (TRIALS x 2) of optimizer `tag` on function fn."""
    key = f"{fn.family}-{fn.i}|{seed}|{OPTIMIZERS[tag]}"
    X = {"GP": gp_ucb, "NM": nelder_mead, "RS": random_search}[tag](key, fn, initial_design(fn.i, seed))
    assert X.shape == (TRIALS, 2), key
    return X
