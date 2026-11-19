"""Shared harness for the contrast screens (figures/09_contrast): problems, teachers, runner.

Problems
  cont   f(x) = bilinear interpolation of a 512 x 512 grid on [0,1]^2 (gp_traj.Fn), optional observation noise
  mixed  x0 in [0,1] continuous, x1 in {0..k-1}: level c is row rows[c] of the base GP grid (linear interpolation in x0)
Base GP function i is identical in every regime and screen (kernel KINDS[i % 4], gp_traj.function_spec(kind, 10000+i)),
the same base functions as figures/08_rf_functions, so all comparisons are paired.

Teachers (all minimise f; regret is always measured on the noise-free function)
  rs                          uniform random search after the shared initial design
  dict(kind="gp", kernel, beta)        corpus GP-UCB teacher, gp_traj.bo_numpy unchanged (noise-free MLE GP)
  dict(kind="gpn", kernel, beta)       same, noise variance chosen by marginal likelihood from NOISES
  dict(kind="tree", model, acq, sd, ...)   tree-ensemble BO (see tree_bo)
  dict(kind="tpe")                     optuna TPE (multivariate), initial design enqueued
  dict(kind="gpord"/"gpcat", kernel, beta)   mixed space only (see mixed_gp_bo)
"""
import json, math, os, sys, time
from pathlib import Path

sys.path.insert(0, os.environ.get("MASTER_THESIS_DIR", "/Users/lucathale-bombien/vs_code/master-thesisv2"))
import gp_traj as G            # sets BLAS threads = 1 before numpy
import gp_np
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, qmc

CFG = G.CFG | G.STUDENT | {"backend": "numpy"}
T = CFG["trials"]
N = 512
_u = np.arange(N) / (N - 1)
GRID_PTS = np.stack(np.meshgrid(_u, _u, indexing="ij"), -1).reshape(-1, 2)
NOISES = (1e-6, 1e-3, 1e-2, 1e-1)
RAW = Path(__file__).resolve().parents[1] / "raw"


# ------------------------------------------------------------------------------------------------ functions
def base(i):
    kind = G.KINDS[i % 4]
    spec = G.function_spec(kind, 10_000 + i, CFG)
    g, _ = G.sample_gp(kind, spec["ls"], spec["var"], spec["seed"], N)
    return spec | {"base": kind, "ls_geo": float(np.exp(np.mean(np.log(spec["ls"]))))}, g


def rf_grid(i, g, n, trees, leaves=None, shuffled=False):
    """RF function exactly as in figures/08_rf_functions (proto.make_fn / proto2.make_fn / rf_data.make)."""
    from sklearn.ensemble import RandomForestRegressor
    r = np.random.default_rng(G.seed_of("rf-fit", i, n))
    X = r.uniform(size=(n, 2))
    seed = int(r.integers(2**31))
    y = G.Fn(g)(X)
    if shuffled:
        y = np.random.default_rng(G.seed_of("shuffle", i)).permutation(y)
    kw = {} if leaves is None and trees > 1 else {"max_leaf_nodes": leaves, "bootstrap": trees > 1}
    rf = RandomForestRegressor(n_estimators=trees, random_state=seed, n_jobs=1, **kw).fit(X, y)
    return rf.predict(GRID_PTS).reshape(N, N)


def iid_grid(i, m):
    """m x m cells of iid U[0,1] values, nearest-neighbour upsampled to the 512 x 512 grid (no local structure below
    the cell size). Same cells for the same (i, m) in every process."""
    c = np.random.default_rng(G.seed_of("iid", i, m)).uniform(size=(m, m))
    idx = np.minimum((_u * m).astype(int), m - 1)
    return c[np.ix_(idx, idx)]


def rosen_fn(i):
    """Random rotated / scaled / shifted 2D Rosenbrock on [0,1]^2, f* = 0 at c.
    f(x) = (1 - u0)^2 + b (u1 - u0^2)^2 with u = (1, 1) + s R (x - c)   (formula as in
    benchmarks/global_optimization_problems: Rosenbrock, a = 1)."""
    r = np.random.default_rng(G.seed_of("rosen", i))
    th = r.uniform(0, 2 * np.pi)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    s = float(np.exp(r.uniform(np.log(2.0), np.log(8.0))))
    c = r.uniform(0.2, 0.8, 2)
    b = float(r.choice([10.0, 100.0]))

    def f(X):
        U = 1.0 + s * (np.atleast_2d(X) - c) @ Rm.T
        return (1 - U[:, 0]) ** 2 + b * (U[:, 1] - U[:, 0] ** 2) ** 2
    hi = float(np.median(f(np.random.default_rng(G.seed_of("rosen-med", i)).uniform(size=(4096, 2)))))
    return f, {"rot": th, "scale": s, "c0": float(c[0]), "c1": float(c[1]), "b": b}, hi


class Prob:
    """A problem instance. true(X) is noise-free; obs(key) returns an observation function for one run.

    Noise is a deterministic function of (run key, point): repeated calls on the same history give the same values
    (the teachers re-evaluate the whole history every iteration), the n-th new point gets the n-th noise draw."""

    def __init__(self, g, space="cont", rows=None, noise=None, fn=None, lo=None, hi=None, floor=-3.0):
        """space="analytic": g is ignored, fn(X) is the exact function, regret is normalised by (lo, hi) as given."""
        self.space, self.noise, self.floor = space, noise, floor
        if space == "analytic":
            self._fn = fn
            self.lo, self.hi = float(lo), float(hi)
            self.sd = float(np.std(fn(np.random.default_rng(0).uniform(size=(4096, 2)))))
            return
        if space == "cont":
            self.g = g.astype(np.float32).astype(float)
            self._fn = G.Fn(self.g)
            vals = self.g
        else:
            self.R = g[np.asarray(rows)].astype(np.float32).astype(float)          # k x 512
            self.k = len(rows)
            vals = self.R
        self.lo, self.hi, self.sd = float(vals.min()), float(vals.max()), float(vals.std())

    def true(self, X):
        X = np.atleast_2d(X)
        if self.space in ("cont", "analytic"):
            return self._fn(X)
        p = np.clip(X[:, 0], 0, 1) * (N - 1)
        j = np.minimum(p.astype(int), N - 2)
        t, c = p - j, X[:, 1].astype(int)
        return (1 - t) * self.R[c, j] + t * self.R[c, j + 1]

    def norm(self, y):
        return (np.asarray(y) - self.lo) / (self.hi - self.lo)

    def obs(self, key):
        if self.noise is None:
            return self.true
        kind, a = self.noise
        r = np.random.default_rng(G.seed_of("noise", key))
        cache = {}

        def f(X):
            X = np.atleast_2d(X)
            out = self.true(X)
            for n, x in enumerate(X):
                kx = (round(float(x[0]), 12), round(float(x[1]), 12))
                if kx not in cache:
                    if kind == "gauss":
                        cache[kx] = a * self.sd * r.standard_normal()
                    else:                                                    # one-sided heavy-tailed outliers
                        u, t2 = r.random(), abs(r.standard_t(2))
                        cache[kx] = 3 * self.sd * t2 if u < a else 0.0
                out[n] += cache[kx]
            return out
        return f

    def sample(self, r, n):
        if self.space != "mixed":
            return r.uniform(size=(n, 2))
        return np.column_stack([r.uniform(size=n), r.integers(0, self.k, n)]).astype(float)


def init_design(prob, i, seed):
    r = np.random.default_rng(G.seed_of("init", str(i), seed))
    n0 = int(r.integers(5, 11))
    if prob.space != "mixed":
        return r.uniform(size=(n0, 2))                                       # identical to figures/08
    return np.column_stack([r.uniform(size=n0), r.integers(0, prob.k, n0)]).astype(float)


# ------------------------------------------------------------------------------------------------ GP teachers
def gp_noise_bo(key, f, X, kernel, beta):
    """bo_numpy with the noise variance chosen by marginal likelihood from NOISES (standardised units)."""
    rng = np.random.default_rng(G.seed_of("fallback", key))
    while len(X) < T:
        y = -f(X)
        z = (y - y.mean()) / max(y.std(ddof=1), 1e-9)
        try:
            best = None
            for nz in NOISES:
                ls, var, ok = gp_np.fit_mle(X, z, kernel, CFG["mle_ls"], CFG["mle_os"], noise=nz)
                v, _ = gp_np.nll(np.log(np.r_[ls, var]), X, z, kernel, nz)
                if best is None or v < best[0]:
                    best = (v, ls, var, nz)
            m = gp_np.GP(X, z, kernel, best[1], best[2], best[3])
            x, _ = gp_np.propose_ucb(m, beta, CFG["restarts"], CFG["raw_samples"],
                                     np.random.default_rng(G.seed_of(key, len(X))))
        except Exception:
            x = rng.uniform(size=2)
        X = np.vstack([X, x])
    return X


class CatGP:
    """Mixed-space GP: Matern(x0) x [rho + (1 - rho) 1(c = c')] (exchangeable / overlap kernel), or for ordinal=True a
    2D ARD Matern on (x0, c / (k - 1)). Hyperparameters by maximum likelihood (multi-start L-BFGS-B, numeric grad)."""

    def __init__(self, X, z, k, kernel, ordinal, noise=1e-6):
        self.X, self.z, self.k, self.kernel, self.ordinal, self.noise = X, z, k, kernel, ordinal, noise
        starts = [np.log([0.25, 0.25, 1.0])] if ordinal else [np.r_[np.log([0.25, 1.0]), s] for s in (-2.0, 0.0, 2.0)]
        lo, hi = np.log(CFG["mle_ls"]), np.log(CFG["mle_os"])
        bnds = [tuple(lo), tuple(lo), tuple(hi)] if ordinal else [tuple(lo), tuple(hi), (-7.0, 7.0)]
        best = None
        for s in starts:
            try:
                r = minimize(self._nll, s, method="L-BFGS-B", bounds=bnds)
                if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                    best = r
            except (np.linalg.LinAlgError, ValueError):
                pass
        self.theta = best.x if best is not None else starts[0]
        K = self._K(X, X, self.theta)
        self.L, _ = gp_np.chol(K + noise * np.eye(len(X)))
        self.alpha = gp_np.cho_solve((self.L, True), z)

    def _K(self, A, B, th):
        if self.ordinal:
            s = 1.0 / (self.k - 1)
            A2, B2 = A * [1, s], B * [1, s]
            return gp_np.kernel(A2, B2, self.kernel, np.exp(th[:2]), math.exp(th[2]))[0]
        k0, _ = gp_np.corr(((A[:, None, 0] - B[None, :, 0]) / math.exp(th[0])) ** 2, self.kernel)
        rho = 1 / (1 + math.exp(-th[2]))
        same = (A[:, None, 1] == B[None, :, 1])
        return math.exp(th[1]) * k0 * np.where(same, 1.0, rho)

    def _nll(self, th):
        K = self._K(self.X, self.X, th)
        L, _ = gp_np.chol(K + self.noise * np.eye(len(K)))
        a = gp_np.cho_solve((L, True), self.z)
        return 0.5 * self.z @ a + np.log(np.diag(L)).sum()

    def predict(self, C):
        Ks = self._K(C, self.X, self.theta)
        mu = Ks @ self.alpha
        V = gp_np.solve_triangular(self.L, Ks.T, lower=True)
        var = self._K(C[:1], C[:1], self.theta)[0, 0] - (V * V).sum(0)
        return mu, np.sqrt(np.maximum(var, 1e-12))


def mixed_gp_bo(key, f, X, k, kernel, beta, ordinal):
    """UCB maximised exactly per level: 512-point x0 grid on every level, then bounded 1D refinement of the 3 best."""
    rng = np.random.default_rng(G.seed_of("fallback", key))
    grid = np.column_stack([np.tile(_u, k), np.repeat(np.arange(k), N)]).astype(float)
    b = math.sqrt(beta)
    while len(X) < T:
        y = -f(X)
        z = (y - y.mean()) / max(y.std(ddof=1), 1e-9)
        try:
            m = CatGP(X, z, k, kernel, ordinal)
            mu, sd = m.predict(grid)
            a = mu + b * sd
            best = grid[int(a.argmax())]
            va = a.max()
            for j in np.argsort(-a)[:3]:
                c = grid[j, 1]
                lo, hi = max(0, grid[j, 0] - 1 / N), min(1, grid[j, 0] + 1 / N)
                r = minimize_scalar(lambda t: -sum(v * w for v, w in zip(m.predict(np.array([[t, c]])), (1, b)))[0],
                                    bounds=(lo, hi), method="bounded")
                if -r.fun > va:
                    best, va = np.array([r.x, c]), -r.fun
            x = best
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            x = np.array([rng.uniform(), rng.integers(k)], float)
        X = np.vstack([X, x])
    return X


# ------------------------------------------------------------------------------------------------ tree teachers
def _features(X, prob):
    if prob.space == "cont":
        return X
    return np.column_stack([X[:, 0], np.eye(prob.k)[X[:, 1].astype(int)]])


def _model(model, n_trees, seed, F, z, msl=1):
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    if model == "rf":
        return RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=msl, min_samples_split=max(2, msl),
                                     random_state=seed, n_jobs=1).fit(F, z)
    if model == "et":
        return ExtraTreesRegressor(n_estimators=n_trees, min_samples_leaf=msl, random_state=seed, n_jobs=1).fit(F, z)
    if model == "bwo":      # Kim & Choi (AISTATS 2022), surrogates.py: 16 copies, 4x oversampled bootstrap, sqrt feats
        return ExtraTreesRegressor(n_estimators=100, max_features="sqrt", bootstrap=True, max_samples=0.25,
                                   random_state=seed, n_jobs=1).fit(np.tile(F, (16, 1)), np.tile(z, 16))
    raise ValueError(model)


def _candidates(X, z, prob, r):
    top = X[np.argsort(-z)[:10]]
    if prob.space == "cont":
        loc = np.repeat(top, 100, 0) + r.choice([.01, .03, .1], (100 * len(top), 1)) * r.standard_normal((100 * len(top), 2))
        return np.vstack([qmc.Sobol(2, scramble=True, seed=r).random(2048), np.clip(loc, 0, 1)])
    k = prob.k
    grid = np.column_stack([np.tile(_u[::4], k), np.repeat(np.arange(k), len(_u[::4]))])
    loc = np.repeat(top, 50, 0)
    loc[:, 0] = np.clip(loc[:, 0] + r.choice([.01, .03, .1], len(loc)) * r.standard_normal(len(loc)), 0, 1)
    swap = np.repeat(top, 20, 0)
    swap[:, 1] = r.integers(0, k, len(swap))
    return np.vstack([grid, loc, swap]).astype(float)


def _logscale(c):
    """SMAC3 RunHistoryLogScaledEncoder (approx.): min-max scale with the minimum pushed below the observed one."""
    lo = c.min() - max(np.percentile(c, 5) - c.min(), 1e-6 * (c.max() - c.min() + 1e-12))
    return np.log((c - lo) / (c.max() - lo))


def tree_bo(key, prob, f, X, model="et", acq="ei", sd="trees", n_trees=50, p_random=0.0, logy=False, agg="mean",
            beta=4.0, kappa=1.0, msl=1):
    """Tree-ensemble BO. Maximises z = standardised -f (or -log-scaled f).
    sd: trees = spread of tree predictions; total = law of total variance (adds within-leaf variance, Hutter et al.
    2014 / Kim & Choi 2022); dist = kappa * 10 * distance to the nearest observation (no tree uncertainty at all)."""
    r = np.random.default_rng(G.seed_of("rfbo", key))
    while len(X) < T:
        if p_random and r.random() < p_random:
            X = np.vstack([X, prob.sample(r, 1)])
            continue
        c = f(X)
        z = -_logscale(c) if logy else -c
        z = (z - z.mean()) / max(z.std(ddof=1), 1e-9)
        m = _model(model, n_trees, int(r.integers(2**31)), _features(X, prob), z, msl)
        C = _candidates(X, z, prob, r)
        FC = _features(C, prob)
        P = np.stack([t.predict(FC) for t in m.estimators_])
        mu = np.median(P, 0) if agg == "median" else P.mean(0)
        if sd == "trees":
            s = P.std(0)
        elif sd == "total":
            V = np.stack([t.tree_.impurity[t.apply(FC.astype(np.float32))] for t in m.estimators_])
            s = np.sqrt(np.maximum((V + P ** 2).mean(0) - P.mean(0) ** 2, 0))
        else:
            d = np.abs(C[:, None, :] - X[None, :, :])
            if prob.space == "mixed":
                d[..., 1] = (d[..., 1] > 0).astype(float)
            s = kappa * 10 * np.sqrt((d ** 2).sum(-1)).min(1)
        s = np.maximum(s, 1e-12)
        if acq == "ei":
            u = (mu - z.max()) / s
            a = s * (u * norm.cdf(u) + norm.pdf(u))
        else:
            a = mu + math.sqrt(beta) * s
        X = np.vstack([X, C[r.choice(np.flatnonzero(a >= a.max() - 1e-12))]])
    return X


def _snap(x, quant):
    x = np.clip(x, 0, 1)
    return np.round(x * 999) / 999 if quant else x


def es_ls(key, f, X, sigma0=0.1, quant=False):
    """(1+1)-ES with the 1/5 success rule, from the best point of the initial design; restarts at a random point when
    the step size collapses. quant: every proposal is snapped to the 1000-level token grid before evaluation."""
    r = np.random.default_rng(G.seed_of("es", key.rsplit("|", 1)[0]))     # same draws for es and es_q (paired)
    y = f(X)
    j = int(y.argmin())
    x, fx, sig = X[j].copy(), float(y[j]), sigma0
    pts = [X]
    n = len(X)
    while n < T:
        cand = _snap(x + sig * r.standard_normal(2), quant)
        fc = float(f(cand[None])[0])
        pts.append(cand[None]); n += 1
        if fc < fx:
            x, fx, sig = cand, fc, sig * 1.5
        else:
            sig *= 1.5 ** -0.25
        if sig < 1e-4:
            x = _snap(r.uniform(size=2), quant)
            if n < T:
                fx = float(f(x[None])[0]); pts.append(x[None]); n += 1
            sig = sigma0
    return np.vstack(pts)[:T]


class _Budget(Exception):
    pass


def nm_ls(key, f, X, size=0.1, quant=False):
    """Nelder-Mead (scipy, bounded) from the best point so far with an initial simplex of edge `size`; restarted from
    the incumbent (fresh simplex, size halved down to 1e-3) whenever it converges, until the budget is spent."""
    r = np.random.default_rng(G.seed_of("nm", key.rsplit("|", 1)[0]))
    pts = [X]
    hist = {"n": len(X), "X": list(X), "y": list(f(X))}

    def obj(x):
        if hist["n"] >= T:
            raise _Budget
        x = _snap(x, quant)
        v = float(f(x[None])[0])
        pts.append(x[None]); hist["n"] += 1; hist["X"].append(x); hist["y"].append(v)
        return v

    sz = size
    while hist["n"] < T:
        x0 = hist["X"][int(np.argmin(hist["y"]))]
        sgn = r.choice([-1.0, 1.0], 2)
        sgn = np.where(x0 + sgn * sz > 1, -1.0, np.where(x0 + sgn * sz < 0, 1.0, sgn))     # stay inside [0,1]^2
        simplex = np.vstack([x0, x0 + np.diag(sgn * sz)])
        try:
            minimize(obj, x0, method="Nelder-Mead", bounds=[(0, 1), (0, 1)],
                     options={"initial_simplex": simplex, "xatol": 1e-4, "fatol": 1e-10, "maxfev": T})
        except _Budget:
            break
        sz = max(sz / 2, 1e-3)
    return np.vstack(pts)[:T]


def tpe_bo(key, prob, f, X):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    s = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=G.seed_of("tpe", key) % 2**31, multivariate=True,
                                                                n_startup_trials=len(X)))
    dist = {"x0": optuna.distributions.FloatDistribution(0, 1),
            "x1": (optuna.distributions.FloatDistribution(0, 1) if prob.space == "cont"
                   else optuna.distributions.CategoricalDistribution(list(range(prob.k))))}
    for x in X:
        p = {"x0": float(x[0]), "x1": float(x[1]) if prob.space == "cont" else int(x[1])}
        s.add_trial(optuna.trial.create_trial(params=p, distributions=dist, value=float(f(np.array([x], float))[0])))
    while len(X) < T:
        t = s.ask(dist)
        x = np.array([[t.params["x0"], float(t.params["x1"])]])
        s.tell(t, float(f(x)[0]))
        X = np.vstack([X, x])
    return X


def run_teacher(name, spec, prob, X0, key):
    f = prob.obs(key)
    if name == "rs":
        r = np.random.default_rng(G.seed_of("rs", key))
        return np.vstack([X0, prob.sample(r, T - len(X0))])
    t = dict(spec)
    kind = t.pop("kind")
    if kind == "gp":
        return G.bo_numpy({"acq": f"ucb{t['beta']}", "mode": "mle", "kernel": t["kernel"], "key": key}, f, X0,
                          CFG | {"trials": T}, {})[0]
    if kind == "gpn":
        return gp_noise_bo(key, f, X0, t["kernel"], t["beta"])
    if kind in ("gpord", "gpcat"):
        return mixed_gp_bo(key, f, X0, prob.k, t["kernel"], t["beta"], kind == "gpord")
    if kind == "tree":
        return tree_bo(key, prob, f, X0, **t)
    if kind == "tpe":
        return tpe_bo(key, prob, f, X0)
    if kind == "es":
        return es_ls(key, f, X0, **t)
    if kind == "nm":
        return nm_ls(key, f, X0, **t)
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------------ runner
def score(prob, X, n0):
    lr = np.log10(np.minimum.accumulate(prob.norm(prob.true(X))) + G.EPS)
    step = [np.sqrt(((X[j, :1] - X[:j, :1]) ** 2).sum(1) + (X[j, 1] != X[:j, 1] if prob.space == "mixed"
                                                            else (X[j, 1] - X[:j, 1]) ** 2)).min() for j in range(n0, len(X))]
    return {"auc_floor": float(np.maximum(lr, prob.floor).mean()), "auc_raw": float(lr.mean()), "final": float(lr[-1]),
            "step_med": float(np.median(step)), "uniq": len({(round(a, 3), round(b, 3)) for a, b in X})}


def run_job(job):
    """job = (regime, i, make, teachers, seeds[, opts]). make(i) -> (spec, Prob, extra-info dict).
    opts: {"T": budget (default CFG trials), "save_traj": "rs" | "all"}. One job runs at a time per worker process,
    so setting the module-level budget T here is safe."""
    global T
    regime, i, make, teachers, seeds = job[:5]
    opts = job[5] if len(job) > 5 else {}
    T = opts.get("T", CFG["trials"])
    save = opts.get("save_traj", "rs")
    spec, prob, info = make(i)
    rows, traj = [], []
    for seed in range(seeds):
        X0 = init_design(prob, i, seed)
        for name, tspec in teachers.items():
            key = f"{regime}-{i}|{seed}|{name}"
            t0 = time.time()
            X = run_teacher(name, tspec, prob, X0, key)
            assert X.shape == (T, 2), (key, X.shape)
            rows.append({"regime": regime, "i": i, "base": spec["base"], "ls_geo": spec["ls_geo"], "seed": seed,
                         "teacher": name, "n0": len(X0), "T": T, **score(prob, X, len(X0)), "sec": time.time() - t0,
                         **info})
            if name == "rs" or save == "all":
                traj.append({"regime": regime, "i": i, "seed": seed, "teacher": name, "T": T, "n0": len(X0),
                             "X": np.round(X, 6).tolist(),
                             "y": np.round(prob.obs(key)(X), 8).tolist()})
    return rows, traj


def run_all(out, jobs, workers=8, shard=None):
    """Runs jobs in a spawn pool; resumable: (regime, i) already present in <out>.jsonl are skipped.
    shard=(k, n): run only every n-th job starting at k, writing <out>.shard<k>.jsonl (combine with merge(out)).
    Trajectories go to <out>_rs_traj.jsonl (RS only, default) or <out>_traj.jsonl (all teachers, opts save_traj=all)."""
    from multiprocessing import get_context
    RAW.mkdir(parents=True, exist_ok=True)
    tag = "" if shard is None else f".shard{shard[0]}"
    if shard is not None:
        jobs = jobs[shard[0]::shard[1]]
    all_traj = bool(jobs) and len(jobs[0]) > 5 and jobs[0][5].get("save_traj") == "all"
    path, tpath = RAW / f"{out}{tag}.jsonl", RAW / f"{out}{'_traj' if all_traj else '_rs_traj'}{tag}.jsonl"
    done = set()
    if path.exists():
        for l in open(path):
            r = json.loads(l)
            done.add((r["regime"], r["i"]))
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    print(f"{out}: {len(jobs)} jobs to run ({len(done)} done)", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(workers) as p, open(path, "a") as fh, open(tpath, "a") as th:
        for n, (rows, traj) in enumerate(p.imap_unordered(run_job, jobs)):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            for r in traj:
                th.write(json.dumps(r) + "\n")
            fh.flush(); th.flush()
            if (n + 1) % max(1, len(jobs) // 20) == 0:
                print(f"  {n + 1}/{len(jobs)}  {(time.time() - t0) / 60:.1f} min", flush=True)


def merge(out):
    """Concatenate <out>.shard*.jsonl (and their trajectory files) into <out>.jsonl / <out>_*traj.jsonl."""
    for suf in ("", "_rs_traj", "_traj"):
        parts = sorted(RAW.glob(f"{out}{suf}.shard*.jsonl"))
        if parts:
            with open(RAW / f"{out}{suf}.jsonl", "w") as fh:
                for q in parts:
                    fh.write(q.read_text())
            print(f"merged {len(parts)} shards -> {out}{suf}.jsonl")


def cli(argv):
    """Strip `--shard k/n` and `--workers w` from argv; returns (positional args, run_all kwargs)."""
    argv, kw = list(argv), {}
    for flag in ("--shard", "--workers"):
        if flag in argv:
            j = argv.index(flag)
            v = argv[j + 1]
            del argv[j:j + 2]
            kw[flag[2:]] = tuple(map(int, v.split("/"))) if flag == "--shard" else int(v)
    return argv, kw
