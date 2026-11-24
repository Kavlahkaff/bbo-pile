"""Student vs teachers on the test functions, paired per (function, seed).
    python -m evaluation.analyze --corpus CORPUS --rollouts r_*.jsonl --output report.txt
mean_lr  mean over trials of log10 normalised regret (+ 1e-8), no floor; main metric with the final win rate
lin      mean linear regret; final  log10 regret after 50 trials; auc_floor3  mean_lr floored at -3 (older results)
Questions: 1 imitation: is the student with tag T as good as teacher T?  2 selection: does BEST / UNK reach the
family's winner, and beat the best single tag, chosen over both families? All differences: negative = first is better."""
import argparse
import glob
import json

import numpy as np
import pandas as pd

from datagen.check_corpus import boot_ci
from datagen.corpus import read
from datagen.optimizers import OPTIMIZERS, WINNER

METRICS = ["mean_lr", "lin", "final", "auc_floor3"]


def row(d, name):
    lr = np.array(d["lr"])
    q = [tuple(p) for p in d["k"]]
    rep = np.mean([q[t] in set(q[:t]) for t in range(d["n_init"], len(q))])
    return {"who": name, "family": d["family"], "group": d["group"], "fid": d["fid"], "seed": d["seed"],
            "mean_lr": lr.mean(), "lin": (10 ** lr).mean(), "final": lr[-1], "auc_floor3": np.maximum(lr, -3).mean(), "repeat": rep}


def diff(R, a, b, metric):
    P = R[R.who.isin([a, b])].pivot_table(index=["fid", "seed"], columns="who", values=metric).dropna()
    d = (P[a] - P[b]).groupby(level=0).mean()
    lo, hi = boot_ci(d)
    return f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]", ((P[a] < P[b]) + .5 * (P[a] == P[b])).mean()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)
    runs = {}
    for p in sorted(x for g in a.rollouts for x in glob.glob(g)):
        for line in open(p):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            runs[d["fid"], d["seed"], d["tag"]] = d
    S = pd.DataFrame([row(d, f"student:{d['tag']}") for d in runs.values()])
    keys = set(zip(S.fid, S.seed))
    T = pd.DataFrame([row(t, f"teacher:{t['opt']}") for t in read(a.corpus, "test") if (t["fid"], t["seed"]) in keys])
    R = pd.concat([S, T], ignore_index=True)
    out = [f"{len(keys)} (function, seed) pairs; runs per arm: {R.groupby('who').size().to_dict()}",
           R.groupby(["family", "who"])[METRICS + ["repeat"]].mean().round(3).to_string()]
    tags = sorted(S.who.unique())
    comps = [(f"student:{o}", f"teacher:{o}", "imitation") for o in OPTIMIZERS if f"student:{o}" in tags]
    for fam, Q in R.groupby("family"):
        out.append(f"\n== {fam} (winner {WINNER[fam]})")
        cs = list(comps)
        for t in ("student:BEST", "student:UNK"):
            if t in tags:
                cs += [(t, f"teacher:{WINNER[fam]}", "selection vs winning teacher")]
                cs += [(t, f"student:{o}", "same model, fixed tag") for o in OPTIMIZERS if f"student:{o}" in tags]
        for x, y, what in cs:
            d, _ = diff(Q, x, y, "mean_lr")
            l, _ = diff(Q, x, y, "lin")
            _, w = diff(Q, x, y, "final")
            out.append(f"  {what:29} {x} - {y}: mean_lr {d}  lin {l}  final win rate {w:.2f}")
    M = R.groupby(["family", "who"]).mean_lr.mean().unstack("who").mean()
    fixed = M[[f"student:{o}" for o in OPTIMIZERS if f"student:{o}" in tags]] if comps else M[[]]
    if len(fixed):
        out.append(f"\nBoth families, equal weight (mean_lr): best single tag {fixed.idxmin()} {fixed.min():.3f}")
        for t in ("student:BEST", "student:UNK"):
            if t in tags:
                out.append(f"  {t}: {M[t]:.3f}, vs best single tag {diff(R, t, fixed.idxmin(), 'mean_lr')[0]}")
    text = "\n".join(out)
    print(text)
    if a.output:
        open(a.output, "w").write(text + "\n")


if __name__ == "__main__":
    main()
