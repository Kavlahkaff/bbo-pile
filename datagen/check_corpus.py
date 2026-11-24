"""Completeness of the corpus and the contrast between the optimizers.
    python -m datagen.check_corpus CORPUS --train 50000
Fails (exit 1) unless every function has every run and each family's winner beats both other optimizers on test and val
functions (mean log10 regret over trials, no floor beyond EPS; CI over functions excludes 0)."""
import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd

from datagen.corpus import read
from datagen.generate import N_TEST, N_VAL, SEEDS
from datagen.optimizers import OPTIMIZERS, WINNER


def boot_ci(d, n=2000, seed=0):
    d = np.asarray(d, float)
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    return np.percentile(d[idx].mean(1), [2.5, 97.5])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--train", type=int, default=50_000)
    a = ap.parse_args(argv)
    n, seen, rows = Counter(), set(), []
    for t in read(a.corpus):
        key = (t["fid"], t["seed"], t["opt"])
        if key in seen:
            sys.exit(f"duplicate run {key}")
        seen.add(key)
        n[t["family"], t["split"], t["opt"]] += 1
        if t["split"] != "train":
            lr = np.array(t["lr"])
            rows.append({"family": t["family"], "group": t["group"], "i": t["i"], "seed": t["seed"], "opt": t["opt"],
                         "mean_lr": lr.mean(), "final": lr[-1], "lin": (10 ** lr).mean(), "auc_floor3": np.maximum(lr, -3).mean()})
    ok = True
    for fam in WINNER:
        for split, size in (("test", N_TEST), ("val", N_VAL), ("train", a.train)):
            for opt in OPTIMIZERS:
                if n[fam, split, opt] != size * SEEDS[split]:
                    print(f"INCOMPLETE {fam} {split} {opt}: {n[fam, split, opt]} of {size * SEEDS[split]}")
                    ok = False
    R = pd.DataFrame(rows)
    print(R.groupby(["family", "opt"])[["mean_lr", "final", "lin", "auc_floor3"]].mean().round(3))
    for fam, w in WINNER.items():
        Q = R[R.family == fam]
        for b in OPTIMIZERS:
            if b != w:
                P = Q.pivot_table(index="i", columns="opt", values="mean_lr")
                lo, hi = boot_ci(P[b] - P[w])
                F = Q.pivot_table(index=["i", "seed"], columns="opt", values="final")
                wr = ((F[w] < F[b]) + .5 * (F[w] == F[b])).mean()
                print(f"{fam}: {w} beats {b} by {(P[b] - P[w]).mean():+.3f} [{lo:+.3f}, {hi:+.3f}], final win rate {wr:.3f}")
                ok &= lo > 0
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
