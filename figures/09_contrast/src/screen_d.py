"""Screen D: exploration level within GP-UCB on plain GP functions.  python screen_d.py FUNCS SEEDS [out]"""
import sys
from common import Prob, base, run_all

TEACHERS = {"rs": None, **{f"m32_b{b}": {"kind": "gp", "kernel": "m32", "beta": b} for b in (0.25, 1, 4, 16)},
            **{f"{k}_b4": {"kind": "gp", "kernel": k, "beta": 4} for k in ("m12", "m52", "rbf")}}


def make(i):
    spec, g = base(i)
    return spec, Prob(g), {}


if __name__ == "__main__":
    funcs, seeds = int(sys.argv[1]), int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else "screen_d"
    run_all(out, [("gp", i, make, TEACHERS, seeds) for i in range(funcs)])
