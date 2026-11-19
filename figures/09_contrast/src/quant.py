"""Quantization analysis: what the student would see of a teacher's trajectory after tokenization.

    python quant.py OUT [OUT ...]          (needs raw/<OUT>_traj.jsonl, i.e. screens run with save_traj="all")

Per regime x teacher, over the trials after the initial design:
  x_dup        share of proposals whose 1000-level token pair equals that of an earlier point (student can't express
               the step at all)
  inv_global   share of true incumbent improvements that leave the incumbent's y token unchanged, y min-max scaled
               over the whole run (corpus default "global")
  (no causal column: with min / max over trials 0..t every new incumbent is written as token 0, exactly like the old
   one, so under "causal" an improvement is never distinguishable from a tie; the share would be 1 by construction)
  inv_log      same as global, but on log(y - min + eps): diagnostic for a log-y encoding (not used in the corpus)
  tie20        share of the last 20 trials whose (global) y token equals the then-incumbent's token
Plus the paired AUC cost of snapping x to the token grid (es_q - es, nm_q - nm) where both teachers are present.
Writes ../quant_<OUT>.csv.
"""
import json, sys
import numpy as np
import pandas as pd
from common import G, RAW

Q = 1000


def quantize(x, lo, hi, q=Q):
    """open_optformer/history.py:quantize, vectorised (round half to even, as Python's round)."""
    if hi == lo:
        return np.zeros_like(np.asarray(x, float), dtype=int)
    return np.rint((np.asarray(x, float) - lo) / (hi - lo) * (q - 1)).astype(int)


def y_tokens(y, mode):
    if mode == "global":
        return quantize(y, y.min(), y.max())
    z = np.log(y - y.min() + 1e-9 * max(y.max() - y.min(), 1e-12))        # "log"
    return quantize(z, z.min(), z.max())


def invisible(y, tok, n0):
    """Share of steps t >= n0 that improve the true incumbent but not its token."""
    inv = imp = 0
    for t in range(max(n0, 1), len(y)):
        j = int(np.argmin(y[:t]))
        if y[t] < y[j]:
            imp += 1
            inv += tok[t] == tok[j]
    return inv / imp if imp else np.nan


def per_run(r):
    X, y, n0 = np.asarray(r["X"]), np.asarray(r["y"], float), r["n0"]
    xt = [tuple(v) for v in np.rint(np.clip(X, 0, 1) * (Q - 1)).astype(int)]
    dup = np.mean([xt[t] in set(xt[:t]) for t in range(n0, len(xt))])
    out = {"x_dup": dup}
    for mode in ("global", "log"):
        out[f"inv_{mode}"] = invisible(y, y_tokens(y, mode), n0)
    tg = y_tokens(y, "global")
    last = range(max(n0, len(y) - 20), len(y))
    out["tie20"] = np.mean([tg[t] == tg[int(np.argmin(y[:t]))] for t in last]) if len(last) else np.nan
    return out


def snap_cost(out):
    R = pd.read_json(RAW / f"{out}.jsonl", lines=True)
    rows = []
    for a in ("es", "nm"):
        if not {a, f"{a}_q"} <= set(R.teacher):
            continue
        for reg, S in R[R.teacher.isin([a, f"{a}_q"])].groupby("regime"):
            P = S.pivot_table(index="i", columns="teacher", values="auc_floor").dropna()   # seeds averaged per function
            d = (P[f"{a}_q"] - P[a]).to_numpy()
            lo, hi = G.boot_ci(d)
            rows.append({"regime": reg, "pair": f"{a}_q - {a}", "mean": round(d.mean(), 4),
                         "ci": f"[{lo:+.4f}, {hi:+.4f}]", "n": len(d)})
    return pd.DataFrame(rows)


def main(outs):
    for out in outs:
        rows = []
        for l in open(RAW / f"{out}_traj.jsonl"):
            r = json.loads(l)
            rows.append({"regime": r["regime"], "teacher": r["teacher"], **per_run(r)})
        D = pd.DataFrame(rows).groupby(["regime", "teacher"]).mean().round(3)
        D.to_csv(RAW.parent / f"quant_{out}.csv")
        pd.set_option("display.width", 200)
        print(f"== {out}: token-level visibility (means over runs)\n{D}\n")
        S = snap_cost(out)
        if len(S):
            print(f"== {out}: AUC cost of snapping x to the token grid (positive = quantized worse)\n{S.to_string(index=False)}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
