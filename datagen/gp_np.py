"""
Pure-numpy GP-BO for small 2D problems: a fast drop-in for the BoTorch path (UCB only).

Why: with <= 50 points in 2D the linear algebra costs microseconds, but every BoTorch acquisition evaluation costs
~2 ms of framework overhead (profiled: 85-90% of a run is optimize_acqf). This module does the same computations
directly: same model, same acquisition, same optimisation scheme, analytic gradients.

Model and optimisation scheme:
  model        zero-mean GP, ARD lengthscales, ScaleKernel(RBF | Matern nu), fixed noise
  kernels      GPyTorch conventions: r = |dx / l|, Matern uses sqrt(2 nu) r
  posterior    latent f (no observation noise), variance clamped at 1e-12 like BoTorch's _mean_and_sigma
  UCB          mu + sqrt(beta) * sigma (BoTorch UpperConfidenceBound)
  optimiser    BoTorch optimize_acqf scheme: `raw` scrambled-Sobol samples, `restarts` starts drawn by Boltzmann
               sampling on standardised acquisition values (eta = 1, best raw point always included), then one joint
               L-BFGS-B over all starts on the summed acquisition (maxiter 2000), return the best start
  MLE          maximises the exact marginal likelihood like fit_gpytorch_mll: sigmoid-transformed raw parameters
               (GPyTorch Interval), loss / n, unbounded L-BFGS-B from initial values (ls 0.25, var 1.0)
Differences that remain: the random streams (Sobol scrambling, Boltzmann draws) and float rounding. numpy's kernel is
exact to ~1e-16; GPyTorch's distance computation loses up to ~1e-7, which shows up as small posterior differences.
"""
import math

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import qmc

KINDS = ["rbf", "m12", "m32", "m52"]
JITTERS = (0.0, 1e-8, 1e-7, 1e-6)          # like linear_operator's psd_safe_cholesky for float64 (absolute jitter)


def corr(r2, kind):
    """Unit-variance correlation k(r2) and its derivative dk/d(r2), r2 = squared scaled distance."""
    if kind == "rbf":
        k = np.exp(-0.5 * r2)
        return k, -0.5 * k
    r = np.sqrt(np.maximum(r2, 0.0))
    if kind == "m12":
        k = np.exp(-r)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(r > 0, -k / (2 * np.where(r > 0, r, 1.0)), 0.0)      # not differentiable at r = 0
        return k, g
    if kind == "m32":
        s = math.sqrt(3.0) * r
        e = np.exp(-s)
        return (1 + s) * e, -1.5 * e
    if kind == "m52":
        s = math.sqrt(5.0) * r
        e = np.exp(-s)
        return (1 + s + s * s / 3) * e, -(5.0 / 6.0) * (1 + s) * e
    raise ValueError(kind)


def _check(X, y=None, kind=None, ls=None, var=None, noise=None):
    """Input validation: fail loudly on anything that would silently produce a wrong model."""
    X = np.asarray(X, float)
    if X.ndim != 2 or len(X) == 0 or not np.isfinite(X).all():
        raise ValueError(f"X must be a finite, non-empty (n x d) array, got shape {X.shape}")
    if y is not None:
        y = np.asarray(y, float)
        if y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError(f"y must be finite with shape ({len(X)},), got {y.shape}")
    if kind is not None and kind not in KINDS:
        raise ValueError(f"unknown kernel {kind!r}")
    if ls is not None and not (np.isfinite(ls).all() and (np.asarray(ls) > 0).all()):
        raise ValueError(f"lengthscales must be finite and > 0, got {ls}")
    if var is not None and not (math.isfinite(var) and var > 0):
        raise ValueError(f"output scale must be finite and > 0, got {var}")
    if noise is not None and not (math.isfinite(noise) and noise >= 0):
        raise ValueError(f"noise must be finite and >= 0, got {noise}")
    return X, y


def kernel(A, B, kind, ls, var):
    """K (m x n), dk/dr2 (m x n, unit variance) and scaled differences D = (a - b) / ls (m x n x d)."""
    D = (A[:, None, :] - B[None, :, :]) / ls
    k, g = corr((D * D).sum(-1), kind)
    return var * k, g, D


def chol(K):
    """Cholesky with escalating diagonal jitter. Returns (L, jitter used)."""
    I = np.eye(len(K))
    for j in JITTERS:
        try:
            return np.linalg.cholesky(K + j * I), j
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("kernel matrix not positive definite")


class GP:
    """Exact GP posterior with fixed hyperparameters (y is the maximisation target, i.e. -f)."""

    def __init__(self, X, y, kind, ls, var, noise):
        X, y = _check(X, y, kind)
        self.X, self.kind = X, kind
        self.ls, self.var = np.broadcast_to(np.asarray(ls, float), (X.shape[1],)).copy(), float(var)
        _check(X, ls=self.ls, var=self.var, noise=float(noise))
        K, _, _ = kernel(X, X, kind, self.ls, self.var)
        self.L, self.jitter = chol(K + noise * np.eye(len(K)))
        self.alpha = cho_solve((self.L, True), y)
        if not np.isfinite(self.alpha).all():
            raise np.linalg.LinAlgError("non-finite GP weights")

    def predict(self, Xs, grad=False):
        """Posterior mean and sd at Xs (m x d); with grad=True also d mean / dx and d sd / dx (m x d)."""
        Xs = np.atleast_2d(Xs)
        Ks, g, D = kernel(Xs, self.X, self.kind, self.ls, self.var)
        mu = Ks @ self.alpha
        V = solve_triangular(self.L, Ks.T, lower=True)
        v = self.var - (V * V).sum(0)
        clamped = v < 1e-12
        sd = np.sqrt(np.where(clamped, 1e-12, v))
        if not grad:
            return mu, sd
        dK = self.var * g[..., None] * 2 * D / self.ls                   # d K(x_i, X_j) / d x_i, (m x n x d)
        W = solve_triangular(self.L.T, V, lower=False)                     # K^-1 Ks^T, (n x m)
        dmu = np.einsum("mnd,n->md", dK, self.alpha)
        dsd = np.einsum("mnd,nm->md", dK, W) * (-1.0 / sd)[:, None]       # d sqrt(v) = -2 w.dK / (2 sd)
        dsd[clamped] = 0.0
        return mu, sd, dmu, dsd


def ucb(gp, Xs, beta, grad=False):
    b = math.sqrt(beta)
    if not grad:
        mu, sd = gp.predict(Xs)
        return mu + b * sd
    mu, sd, dmu, dsd = gp.predict(Xs, grad=True)
    return mu + b * sd, dmu + b * dsd


def propose_ucb(gp, beta, restarts, raw, rng, eta=1.0):
    """Maximise UCB on [0,1]^d with BoTorch's optimize_acqf scheme. Returns (x, acq value).

    Guarantees: x lies in [0,1]^d and is finite, and its UCB value is at least that of every start point (the joint
    L-BFGS-B over all restarts minimises the *sum*, which could lower an individual start; the starts are kept as
    candidates). Raises FloatingPointError if the acquisition is not finite on the raw samples."""
    if not (restarts >= 1 and raw >= restarts):
        raise ValueError(f"need 1 <= restarts <= raw, got restarts={restarts}, raw={raw}")
    d = gp.X.shape[1]
    R = qmc.Sobol(d, scramble=True, seed=rng).random(raw)
    a = ucb(gp, R, beta)
    if not np.isfinite(a).all():
        raise FloatingPointError("non-finite acquisition values on raw samples")
    if a.std() > 0:
        w = np.exp(eta * (a - a.mean()) / a.std())
        idx = rng.choice(raw, size=restarts, replace=False, p=w / w.sum())
    else:
        idx = rng.choice(raw, size=restarts, replace=False)
    best = int(a.argmax())
    if best not in idx:
        idx[-1] = best

    def f(z):
        v, dv = ucb(gp, z.reshape(restarts, d), beta, grad=True)
        return -v.sum(), -dv.ravel()

    S = R[idx]
    try:
        res = minimize(f, S.ravel(), jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * (restarts * d),
                       options={"maxiter": 2000})
        Z = np.vstack([np.clip(res.x.reshape(restarts, d), 0.0, 1.0), S])
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        Z = S                                                                  # optimiser failed: best start point
    v = ucb(gp, Z, beta)
    v = np.where(np.isfinite(v) & np.isfinite(Z).all(1), v, -np.inf)
    j = int(v.argmax())
    if not np.isfinite(v[j]):
        raise FloatingPointError("no finite acquisition value after optimisation")
    return Z[j].copy(), float(v[j])


def nll(theta, X, y, kind, noise):
    """Negative log marginal likelihood and gradient in theta = (log ls_1..d, log var)."""
    d = X.shape[1]
    ls, var = np.exp(theta[:d]), math.exp(theta[d])
    K, g, D = kernel(X, X, kind, ls, var)
    n = len(y)
    L, j = chol(K + noise * np.eye(n))
    alpha = cho_solve((L, True), y)
    val = 0.5 * y @ alpha + np.log(np.diag(L)).sum() + 0.5 * n * math.log(2 * math.pi)
    A = np.outer(alpha, alpha) - cho_solve((L, True), np.eye(n))           # d nll / d theta = -0.5 tr(A dK)
    grad = np.empty(d + 1)
    for i in range(d):
        dK = var * g * (-2.0 * D[..., i] ** 2)                              # d K / d log ls_i
        grad[i] = -0.5 * (A * dK).sum()
    grad[d] = -0.5 * (A * K).sum()                                          # d K / d log var = K
    return val, grad


def fit_mle(X, y, kind, ls_bounds, var_bounds, ls0=0.25, var0=1.0, noise=1e-6):
    """Maximum-likelihood ls / var within bounds. Returns (ls array, var, success flag).

    Same optimisation problem *and path* as BoTorch's fit_gpytorch_mll on the same model: GPyTorch Interval constraints
    parameterise value = lo + (hi - lo) * sigmoid(raw), and L-BFGS-B (scipy defaults, unbounded) minimises the
    negative marginal log likelihood divided by n from the same initial values. The marginal likelihood of RBF and
    Matern-5/2 models is often multimodal, so matching the path (not just the objective) matters."""
    X, y = _check(X, y, kind)
    n, d = X.shape
    lo = np.array([ls_bounds[0]] * d + [var_bounds[0]], float)
    hi = np.array([ls_bounds[1]] * d + [var_bounds[1]], float)
    v0 = np.array([ls0] * d + [var0], float)
    if not (0 < lo).all() or not (lo < hi).all() or not ((lo <= v0) & (v0 <= hi)).all():
        raise ValueError(f"need 0 < lower < upper and initial values inside the bounds, got {lo}, {hi}, {v0}")
    p0 = np.clip((v0 - lo) / (hi - lo), 1e-12, 1 - 1e-12)
    raw0 = np.log(p0) - np.log1p(-p0)                                       # inverse sigmoid

    def value(raw):
        sig = expit(raw)                                                    # no overflow for large |raw|
        return lo + (hi - lo) * sig, (hi - lo) * sig * (1.0 - sig)

    def obj(raw):
        v, dv = value(raw)
        f, g = nll(np.log(v), X, y, kind, noise)                            # g = d nll / d log v
        return f / n, g / v * dv / n

    # BoTorch semantics: scipy status 0 (converged) or 1 (iteration limit) is a success; anything else (abnormal line
    # search, non-finite values, exceptions) rolls the parameters back to the initial values, like fit_gpytorch_mll
    # followed by a rollback. (BoTorch retries 5x, but without priors every retry starts from the same values.)
    try:
        res = minimize(obj, raw0, jac=True, method="L-BFGS-B")
        v = value(res.x)[0]
        ok = bool(res.status in (0, 1) and np.isfinite(res.fun) and np.isfinite(v).all())
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        ok = False
    if not ok:
        v = v0
    return v[:d].copy(), float(v[d]), ok
