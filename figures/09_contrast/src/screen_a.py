"""Screen A: tuned tree teachers on plain GP and RF functions.
    python screen_a.py FUNCS SEEDS [out] [teacher,teacher,...] [--shard k/n] [--workers w]
Scale-up run (2026-11-19): screen_a.py 400 30 screen_a_big rs,m32,et_ei,et_dist0.5,smac3_trees"""
import sys
from common import Prob, base, cli, rf_grid, run_all

REGIMES = {"gp": None, "n50_t5": dict(n=50, trees=5), "n50_t5_shuffled": dict(n=50, trees=5, shuffled=True),
           "n200_t1_L16": dict(n=200, trees=1, leaves=16)}
SMAC = dict(kind="tree", model="rf", n_trees=10, msl=3, logy=True, p_random=0.2, acq="ei")
TEACHERS = {"rs": None, "m32": {"kind": "gp", "kernel": "m32", "beta": 4},
            "et_ei": dict(kind="tree", model="et", acq="ei", sd="trees"),
            "smac3_trees": SMAC | {"sd": "trees"}, "smac3_total": SMAC | {"sd": "total"},
            "bwo": dict(kind="tree", model="bwo", acq="ei", sd="total"),
            **{f"et_dist{k}": dict(kind="tree", model="et", acq="ucb", beta=4, sd="dist", kappa=k) for k in (0.25, 0.5, 1, 2)},
            **{f"et_ucb{b}": dict(kind="tree", model="et", acq="ucb", beta=b, sd="trees") for b in (4, 16, 64)}}


def make_regime(regime):
    def make(i):
        spec, g = base(i)
        if REGIMES[regime] is not None:
            g = rf_grid(i, g, **REGIMES[regime])
        return spec, Prob(g), {}
    return make


def make_gp(i): return make_regime("gp")(i)
def make_n50_t5(i): return make_regime("n50_t5")(i)
def make_n50_t5_shuffled(i): return make_regime("n50_t5_shuffled")(i)
def make_n200_t1_L16(i): return make_regime("n200_t1_L16")(i)
MAKE = {"gp": make_gp, "n50_t5": make_n50_t5, "n50_t5_shuffled": make_n50_t5_shuffled, "n200_t1_L16": make_n200_t1_L16}

if __name__ == "__main__":
    sys.argv, kw = cli(sys.argv)
    funcs, seeds = int(sys.argv[1]), int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else "screen_a"
    teach = TEACHERS if len(sys.argv) <= 4 else {k: TEACHERS[k] for k in sys.argv[4].split(",")}
    run_all(out, [(r, i, MAKE[r], teach, seeds) for i in range(funcs) for r in REGIMES], **kw)
