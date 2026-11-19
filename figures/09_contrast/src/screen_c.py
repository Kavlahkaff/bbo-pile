"""Screen C: observation noise and outliers on plain GP functions (regret on the noise-free function).
    python screen_c.py FUNCS SEEDS [out] [teacher,teacher,...]"""
import sys
from common import Prob, base, run_all

REGIMES = {"clean": None, "gauss0.05": ("gauss", 0.05), "gauss0.2": ("gauss", 0.2), "outlier0.1": ("outlier", 0.1)}
TREE = dict(kind="tree", model="et", acq="ucb", beta=4, sd="dist", kappa=0.5)
TEACHERS = {"rs": None, "m32": {"kind": "gp", "kernel": "m32", "beta": 4},
            "m32n": {"kind": "gpn", "kernel": "m32", "beta": 4},
            "tree": TREE, "tree_median": TREE | {"agg": "median"},
            "smac3_trees": dict(kind="tree", model="rf", n_trees=10, msl=3, logy=True, p_random=0.2, acq="ei", sd="trees"),
            "tpe": {"kind": "tpe"}}


def make_any(i, regime):
    spec, g = base(i)
    return spec, Prob(g, noise=REGIMES[regime]), {}


def make_clean(i): return make_any(i, "clean")
def make_g05(i): return make_any(i, "gauss0.05")
def make_g2(i): return make_any(i, "gauss0.2")
def make_out(i): return make_any(i, "outlier0.1")
MAKE = {"clean": make_clean, "gauss0.05": make_g05, "gauss0.2": make_g2, "outlier0.1": make_out}

if __name__ == "__main__":
    funcs, seeds = int(sys.argv[1]), int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else "screen_c"
    teach = TEACHERS if len(sys.argv) <= 4 else {k: TEACHERS[k] for k in sys.argv[4].split(",")}
    run_all(out, [(r, i, MAKE[r], teach, seeds) for i in range(funcs) for r in MAKE])
