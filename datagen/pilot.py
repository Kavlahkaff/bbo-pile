"""Contrast pilot: does the expected optimizer win clearly on every family?
    python -m datagen.generate --out PILOT --pilot 1000 ... && python -m datagen.pilot PILOT --funcs 1000
Unit = function (seeds averaged first), CIs = bootstrap over functions. No regret floor beyond EPS. Rule, fixed before the run:
  winner     for each family, WINNER against EACH other optimizer: final win rate (ties 1/2) lower CI >= 0.65,
             and mean log10 regret better by >= 0.10 with a CI above 0
  subgroups  in every subgroup the winner's mean log10 regret is better than each other optimizer's, CI above 0
A family passes if both hold. GO = all families pass. Trained families that fail should be dropped, not retuned.
Reported, not in the rule: linear regret, floor -3, and whether random points identify the winning side (gp vs the others),
including a held-out family the classifier never saw."""
import argparse
import sys

import numpy as np
import pandas as pd

from datagen.check_corpus import boot_ci
from datagen.corpus import read
from datagen.functions import FAMILIES, HELD_OUT
from datagen.generate import SEEDS
from datagen.optimizers import OPTIMIZERS, WINNER


def ci(x):
    x = np.asarray(x, float)
    lo, hi = boot_ci(x)
    return x.mean(), lo, hi


def fmt(c):
    return f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"


def lead(Q, w, b):
    P = Q.pivot_table(index="i", columns="opt", values="mean_lr")
    return ci(P[b] - P[w])


def win(Q, w, b):
    F = Q.pivot_table(index=["i", "seed"], columns="opt", values="final")
    return ci(((F[w] < F[b]) + .5 * (F[w] == F[b])).groupby(level=0).mean())


def features(X, y):
    """Affine-invariant shape of a few random evaluations: y quantiles and variogram of min-max scaled y and of its rank."""
    z = (y - y.min()) / (np.ptp(y) + 1e-12)
    rank = np.argsort(np.argsort(z)) / (len(z) - 1)
    d = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))[np.triu_indices(len(y), 1)]
    out = np.quantile(z, np.linspace(.1, .9, 9)).tolist()
    for v in (z, rank):
        sq = ((v[:, None] - v[None]) ** 2)[np.triu_indices(len(y), 1)]
        out += [sq[(d >= a) & (d < b)].mean() if ((d >= a) & (d < b)).any() else sq.mean()
                for a, b in ((0, .15), (.15, .3), (.3, .5), (.5, 2))]
    return out


def identify(T):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    fam = np.array([t["family"] for t in T])
    side, idx = np.array([WINNER[f] for f in fam]), np.array([t["i"] for t in T])
    w = np.where(fam == "gp", 3.0, 1.0)                              # both sides equally heavy
    tr = ~np.isin(fam, list(HELD_OUT))
    for k in (10, 20):
        F = np.array([features(np.array(t["k"][:k]) / 999, np.array(t["y"][:k])) for t in T])
        clf = RandomForestClassifier(200, min_samples_leaf=2, random_state=0, n_jobs=-1)
        p = cross_val_predict(clf, F[tr], side[tr], groups=idx[tr], cv=GroupKFold(5), params={"sample_weight": w[tr]})
        acc = {f: round(float((p[fam[tr] == f] == WINNER[f]).mean()), 3) for f in FAMILIES if f not in HELD_OUT}
        clf.fit(F[tr], side[tr], sample_weight=w[tr])
        held = {f: round(float((clf.predict(F[fam == f]) == WINNER[f]).mean()), 3) for f in sorted(HELD_OUT)}
        print(f"  {k} random points -> correct winner: trained families {acc}, held out {held}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pilot")
    ap.add_argument("--funcs", type=int, default=1000)
    a = ap.parse_args(argv)
    rows, rs, seen = [], [], set()
    for t in read(a.pilot):
        key = (t["fid"], t["seed"], t["opt"])
        if key in seen:
            sys.exit(f"duplicate run {key}")
        seen.add(key)
        lr = np.array(t["lr"])
        rows.append({k: t[k] for k in ("family", "group", "i", "seed", "opt")}
                    | {"mean_lr": lr.mean(), "final": lr[-1], "lin": (10 ** lr).mean(), "floor3": np.maximum(lr, -3).mean(),
                       "solved1e-3": float(lr[-1] < -3)})
        if t["opt"] == "RS" and t["seed"] == 0:
            rs.append(t)
    R = pd.DataFrame(rows)
    n = R.groupby(["family", "opt"]).size()
    for f in FAMILIES:
        for o in OPTIMIZERS:
            if n.get((f, o), 0) != a.funcs * SEEDS["pilot"]:
                sys.exit(f"INCOMPLETE {f} {o}: {n.get((f, o), 0)} of {a.funcs * SEEDS['pilot']} runs")
    print(R.groupby(["family", "opt"])[["mean_lr", "final", "lin", "floor3", "solved1e-3"]].mean().round(3).to_string())
    verdict = {}
    for f in FAMILIES:
        Q, w, ok = R[R.family == f], WINNER[f], True
        print(f"\n{f} (expected winner {w}{', held out' if f in HELD_OUT else ''})")
        for b in OPTIMIZERS:
            if b == w:
                continue
            d, r = lead(Q, w, b), win(Q, w, b)
            good = r[1] >= .65 and d[0] >= .10 and d[1] > 0
            ok &= good
            P = Q.pivot_table(index="i", columns="opt", values=["lin", "floor3", "mean_lr"])
            print(f"  {w} vs {b}: final win rate {fmt(r)}, lead {fmt(d)}  {'ok' if good else 'NO'}   (linear "
                  f"{(P['lin'][b] - P['lin'][w]).mean():+.4f}, floor -3 {(P['floor3'][b] - P['floor3'][w]).mean():+.3f}, "
                  f"better on {(P['mean_lr'][w] < P['mean_lr'][b]).mean():.0%} of functions)")
            for g, q in Q.groupby("group"):
                s = lead(q, w, b)
                ok &= s[1] > 0
                print(f"      {g:7} n={q.i.nunique():4}: {fmt(s)}  {'' if s[1] > 0 else 'NO'}")
        verdict[f] = bool(ok)
    print("\nIdentifiability (reported only)")
    identify(rs)
    print(f"\nVERDICT {'GO' if all(verdict.values()) else 'NO-GO'}: {verdict}")


if __name__ == "__main__":
    main()
