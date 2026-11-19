"""Screen B: mixed continuous x categorical 2D. Ordinal vs nominal level order, k levels.
    python screen_b.py FUNCS SEEDS [out] [teacher,teacher,...]"""
import sys
from common import G, N, Prob, base, run_all
import numpy as np

KS = (8, 16, 32)
TREES = {"et_ei": dict(kind="tree", model="et", acq="ei", sd="trees"),
         "smac3_trees": dict(kind="tree", model="rf", n_trees=10, msl=3, logy=True, p_random=0.2, acq="ei", sd="trees"),
         **{f"et_dist{k}": dict(kind="tree", model="et", acq="ucb", beta=4, sd="dist", kappa=k) for k in (0.25, 0.5)}}
TEACHERS = {"rs": None, "gpord": {"kind": "gpord", "kernel": "m32", "beta": 4},
            "gpcat": {"kind": "gpcat", "kernel": "m32", "beta": 4}, "tpe": {"kind": "tpe"}, **TREES}


def levels(i, k, order):
    rows = np.round(np.arange(k) * (N - 1) / (k - 1)).astype(int)
    if order == "ord":
        return rows
    r = np.random.default_rng(G.seed_of("levels", i, k))
    while True:                                   # identity and reversal keep every adjacency: redraw
        p = r.permutation(k)
        if not ((p == np.arange(k)).all() or (p == np.arange(k)[::-1]).all()):
            return rows[p]


def make_any(i, k, order):
    spec, g = base(i)
    rows = levels(i, k, order)
    prob = Prob(g, space="mixed", rows=rows)
    adj = float(np.mean([np.corrcoef(prob.R[c], prob.R[c + 1])[0, 1] for c in range(k - 1)]))
    return spec, prob, {"k": k, "order": order, "adj_corr": adj, "lo": prob.lo, "hi": prob.hi}


def make_k8_ord(i): return make_any(i, 8, "ord")
def make_k8_nom(i): return make_any(i, 8, "nom")
def make_k16_ord(i): return make_any(i, 16, "ord")
def make_k16_nom(i): return make_any(i, 16, "nom")
def make_k32_ord(i): return make_any(i, 32, "ord")
def make_k32_nom(i): return make_any(i, 32, "nom")
MAKE = {f"k{k}_{o}": globals()[f"make_k{k}_{o}"] for k in KS for o in ("ord", "nom")}

if __name__ == "__main__":
    funcs, seeds = int(sys.argv[1]), int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else "screen_b"
    teach = TEACHERS if len(sys.argv) <= 4 else {k: TEACHERS[k] for k in sys.argv[4].split(",")}
    run_all(out, [(r, i, MAKE[r], teach, seeds) for i in range(funcs) for r in MAKE])
