"""Screen F: GP-BO vs local search on random rotated/scaled Rosenbrock functions, plus x-quantized local search.
    python screen_f.py FUNCS SEEDS [out] [teacher,...] [--shard k/n] [--workers w]
Regimes: rosen_T50 / rosen_T200 (budget 50 = corpus, 200 = secondary arm) and gp_T50 / gp_T200 (plain corpus GP).
Rosenbrock regret is normalised by (0, median of f on uniform points), floor -6 (the raw range would saturate at -3).
es_q / nm_q snap every proposal to the 1000-level token grid; their random draws are paired with es / nm."""
import sys
from common import Prob, base, cli, rosen_fn, run_all

TEACHERS = {"rs": None, "m32": {"kind": "gp", "kernel": "m32", "beta": 4},
            "es": {"kind": "es"}, "nm": {"kind": "nm"},
            "es_q": {"kind": "es", "quant": True}, "nm_q": {"kind": "nm", "quant": True}}
BUDGETS = (50, 200)


def make_gp(i):
    spec, g = base(i)
    return spec, Prob(g), {}


def make_rosen(i):
    spec, _ = base(i)
    f, info, hi = rosen_fn(i)
    return spec, Prob(None, space="analytic", fn=f, lo=0.0, hi=hi, floor=-6.0), info


MAKE = {"gp": make_gp, "rosen": make_rosen}

if __name__ == "__main__":
    sys.argv, kw = cli(sys.argv)
    funcs, seeds = int(sys.argv[1]), int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else "screen_f"
    teach = TEACHERS if len(sys.argv) <= 4 else {k: TEACHERS[k] for k in sys.argv[4].split(",")}
    run_all(out, [(f"{r}_T{t}", i, MAKE[r], teach, seeds, {"T": t, "save_traj": "all"})
                  for t in BUDGETS for i in range(funcs) for r in MAKE], **kw)
