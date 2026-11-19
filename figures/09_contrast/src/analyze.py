"""Analysis for the contrast screens: per-family teacher means, crossing test, CV headroom, early identifiability.

    python analyze.py screen_d [--family kind_ls|ls|kind|regime] [--pool t1,t2,...] [--ref m32_b4]

Unit of analysis: a function = (regime, i), seeds averaged first. CIs: paired bootstrap over functions (gp_traj.boot_ci).
Pass rule (pre-registered in the plan):
  1 crossing   >= 2 teachers are each best in >= 1 family with the paired 95% CI vs the runner-up excluding 0
  2 headroom   best-per-family minus best-fixed >= 0.10 floored AUC, CI excluding 0; choices made on half the base
               functions, scored on the other half (30 random splits, grouped by base index i)
  3 early      family predictable from the first 10 / 20 RS observations; learned choice beats the fixed teacher
  4 sanity     every pool teacher beats RS on average
"""
import argparse, json, sys
import numpy as np
import pandas as pd
from common import G, RAW

FLOOR_HEADROOM = 0.10


def load(out):
    R = pd.read_json(RAW / f"{out}.jsonl", lines=True)
    return R


def add_family(R, how):
    if how == "regime":
        R["family"] = R.regime
    elif how in ("kind_ls", "ls"):
        f = R.groupby("i").ls_geo.first()
        t = pd.qcut(f, 3, labels=["ls-short", "ls-mid", "ls-long"])
        R["lsbin"] = R.i.map(t).astype(str)
        R["family"] = (R.base + "|" + R.lsbin) if how == "kind_ls" else R.lsbin
    elif how == "kind":
        R["family"] = R.base
    else:
        raise ValueError(how)
    return R


def per_function(R, pool):
    P = R[R.teacher.isin(pool)].pivot_table(index=["family", "regime", "i"], columns="teacher", values="auc_floor")
    return P[pool].dropna()


def family_table(P, ref):
    rows = []
    for fam, Q in P.groupby(level=0):
        m = Q.mean()
        order = m.sort_values().index
        b, r2 = order[0], order[1]
        d = (Q[b] - Q[r2]).to_numpy()
        lo, hi = G.boot_ci(d)
        rows.append({"family": fam, "n": len(Q), **m.round(3).to_dict(), "best": b, "runner_up": r2,
                     "best-runner": round(d.mean(), 3), "ci": f"[{lo:+.3f}, {hi:+.3f}]", "sig": hi < 0,
                     f"best-{ref}": round((Q[b] - Q[ref]).mean(), 3)})
    return pd.DataFrame(rows).set_index("family")


def headroom(P, n_splits=30, seed=0):
    """CV value of choosing the teacher per family vs one fixed teacher. Returns (mean, ci, per-function diffs)."""
    rng = np.random.default_rng(seed)
    fam = P.index.get_level_values(0).to_numpy()
    base_i = P.index.get_level_values(2).to_numpy()
    V, cols = P.to_numpy(), list(P.columns)
    ids = np.unique(base_i)
    acc = np.zeros(len(P)); cnt = np.zeros(len(P))
    picks_fixed, picks_fam = {}, {}
    for _ in range(n_splits):
        tr_ids = rng.choice(ids, len(ids) // 2, replace=False)
        tr = np.isin(base_i, tr_ids)
        jf = int(V[tr].mean(0).argmin())
        picks_fixed[cols[jf]] = picks_fixed.get(cols[jf], 0) + 1
        jfam = {f: int(V[tr & (fam == f)].mean(0).argmin()) for f in np.unique(fam)}
        for f, j in jfam.items():
            picks_fam.setdefault(f, {}).setdefault(cols[j], 0)
            picks_fam[f][cols[j]] += 1
        te = ~tr
        choice = np.array([jfam[f] for f in fam[te]])
        acc[te] += V[te][:, jf] - V[te][np.arange(te.sum()), choice]
        cnt[te] += 1
    d = acc[cnt > 0] / cnt[cnt > 0]
    lo, hi = G.boot_ci(d)
    return d.mean(), (lo, hi), picks_fixed, picks_fam


# ---------------------------------------------------------------------------------------------- identifiability
def features(X, y, space, k=None):
    X, y = np.asarray(X, float), np.asarray(y, float)
    if space == "cont":
        return G.early_features(X, y)
    Xs = X.copy(); Xs[:, 1] = X[:, 1] / (k - 1)
    f = G.early_features(Xs, y)
    z = (y - y.mean()) / (y.std() + 1e-12)
    dl = np.abs(X[:, None, 1] - X[None, :, 1]); dx = np.abs(X[:, None, 0] - X[None, :, 0])
    iu = np.triu_indices(len(y), 1)
    sq = ((z[:, None] - z[None]) ** 2)[iu]
    adj = sq[(dl[iu] == 1) & (dx[iu] < 0.15)]
    far = sq[(dl[iu] >= 2) & (dx[iu] < 0.15)]
    return f + [adj.mean() if len(adj) else 2.0, far.mean() if len(far) else 2.0,
                (adj.mean() if len(adj) else 2.0) - (far.mean() if len(far) else 2.0)]


def identifiability(out, R, P, ks=(10, 20), folds=5, seed=0):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    if (RAW / f"{out}_rs_traj.jsonl").exists():
        T = pd.read_json(RAW / f"{out}_rs_traj.jsonl", lines=True)
    else:                                                      # screens run with save_traj="all" (E, F)
        T = pd.read_json(RAW / f"{out}_traj.jsonl", lines=True)
        T = T[T.teacher == "rs"].drop(columns=["teacher", "T", "n0"], errors="ignore")
    T = T[T.regime.isin(R.regime.unique())]
    space = "mixed" if "k" in R.columns and R.k.notna().any() else "cont"
    fam_of = R.groupby(["regime", "i"]).family.first()
    kk = R.groupby(["regime", "i"]).k.first() if space == "mixed" else None
    seedP = R[R.teacher.isin(P.columns)].pivot_table(index=["regime", "i", "seed"], columns="teacher",
                                                     values="auc_floor")[list(P.columns)].dropna()
    T = T.set_index(["regime", "i", "seed"]).loc[seedP.index.intersection(T.set_index(["regime", "i", "seed"]).index)]
    seedP = seedP.loc[T.index]
    fam = np.array([fam_of[(r, i)] for r, i, _ in T.index])
    base_i = np.array([i for _, i, _ in T.index])
    V = seedP.to_numpy()
    Vf = seedP.groupby(level=[0, 1]).transform("mean").to_numpy()
    rng = np.random.default_rng(seed)
    ids = np.unique(base_i); rng.shuffle(ids)
    fold = {i: n % folds for n, i in enumerate(ids)}
    fo = np.array([fold[i] for i in base_i])
    lines = []
    for k in ks:
        F = np.array([features(np.asarray(x)[:k], np.asarray(y)[:k], space,
                               None if kk is None else int(kk[(r, i)])) for (r, i, _), x, y in zip(T.index, T.X, T.y)])
        acc, learned, fixed = [], np.zeros(len(F)), np.zeros(len(F))
        for q in range(folds):
            tr, te = fo != q, fo == q
            clf = RandomForestClassifier(300, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(F[tr], fam[tr])
            acc.append((clf.predict(F[te]) == fam[te]).mean())
            reg = RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(F[tr], Vf[tr])
            j = reg.predict(F[te]).argmin(1)
            learned[te] = V[te][np.arange(te.sum()), j]
            fixed[te] = V[te][:, int(V[tr].mean(0).argmin())]
        d = pd.Series(learned - fixed).groupby(base_i).mean().to_numpy()
        lo, hi = G.boot_ci(d)
        chance = pd.Series(fam).value_counts(normalize=True).max()
        lines.append(f"  k={k:>2}: family accuracy {np.mean(acc):.2f} (majority {chance:.2f}); learned - fixed "
                     f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--family", default="regime")
    ap.add_argument("--pool", default=None, help="teachers in the selection pool (default: all but rs)")
    ap.add_argument("--ref", default="m32")
    ap.add_argument("--regimes", default=None)
    ap.add_argument("--no-ident", action="store_true")
    a = ap.parse_args(argv)
    R = load(a.out)
    if a.regimes:
        R = R[R.regime.isin(a.regimes.split(","))].copy()
    R = add_family(R, a.family)
    teachers = sorted(R.teacher.unique())
    pool = a.pool.split(",") if a.pool else [t for t in teachers if t != "rs"]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(f"== {a.out}: {R.groupby(['regime', 'i']).ngroups} functions, {R.seed.nunique()} seeds, family = {a.family}")
    allP = per_function(R, teachers)
    print("\nMean floored AUC per family (lower = better), all teachers:")
    print(allP.groupby(level=0).mean().round(3).to_string())
    if "step_med" in R:
        print("\nMedian step distance (new point to nearest earlier point), per teacher:")
        print(R.groupby("teacher").step_med.median().round(3).to_string())
    P = per_function(R, pool)
    ref = a.ref if a.ref in pool else pool[0]
    print(f"\nPool: {pool}")
    ft = family_table(P, ref)
    print(ft[["n", "best", "runner_up", "best-runner", "ci", "sig", f"best-{ref}"]].to_string())
    winners = set(ft.best[ft.sig])
    h, (lo, hi), pf, pfam = headroom(P)
    rs = allP["rs"] if "rs" in allP else None
    worse_rs = [t for t in pool if t != "rs" and rs is not None and allP[t].mean() >= rs.mean()]
    print(f"\n1 crossing: significant family winners = {sorted(winners)} -> {'PASS' if len(winners) >= 2 else 'fail'}")
    print(f"2 headroom: per-family choice beats fixed by {h:+.3f} [{lo:+.3f}, {hi:+.3f}] -> "
          f"{'PASS' if h >= FLOOR_HEADROOM and lo > 0 else 'fail'}   (fixed picks {pf})")
    print("   per-family picks:", {f: max(v, key=v.get) for f, v in pfam.items()})
    print(f"4 sanity: teachers not beating RS: {worse_rs or 'none'} -> {'PASS' if not worse_rs else 'fail'}")
    if not a.no_ident:
        print("3 early identifiability (5-fold CV grouped by base function):")
        print("\n".join(identifiability(a.out, R, P)))


if __name__ == "__main__":
    main()
