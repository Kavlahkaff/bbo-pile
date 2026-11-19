"""Screen E: functions without local structure (iid U[0,1] cells), where random search should not lose.
    python screen_e.py FUNCS SEEDS [out] [teacher,...] [--shard k/n] [--workers w]
Regimes: iid_m{8,32,128,512} = m x m cells of iid uniform values (nearest neighbour), plus the plain corpus GP as control.
Hypothesis: RS >= BO on m128 / m512, BO >> RS on gp. All trajectories are saved (for quant.py)."""
import sys
from common import Prob, base, cli, iid_grid, run_all

MS = (8, 32, 128, 512)
TEACHERS = {"rs": None, "m32": {"kind": "gp", "kernel": "m32", "beta": 4},
            "m12": {"kind": "gp", "kernel": "m12", "beta": 4},
            "es": {"kind": "es"}, "nm": {"kind": "nm"},
            "et_dist0.5": dict(kind="tree", model="et", acq="ucb", beta=4, sd="dist", kappa=0.5)}


def make_gp(i):
    spec, g = base(i)
    return spec, Prob(g), {}


def _make_iid(m):
    def make(i):
        spec, _ = base(i)                      # keeps base / ls_geo columns; the function itself is pure noise
        return spec, Prob(iid_grid(i, m)), {"m": m}
    return make


def make_m8(i): return _make_iid(8)(i)
def make_m32(i): return _make_iid(32)(i)
def make_m128(i): return _make_iid(128)(i)
def make_m512(i): return _make_iid(512)(i)
MAKE = {"gp": make_gp, "iid_m8": make_m8, "iid_m32": make_m32, "iid_m128": make_m128, "iid_m512": make_m512}

if __name__ == "__main__":
    sys.argv, kw = cli(sys.argv)
    funcs, seeds = int(sys.argv[1]), int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else "screen_e"
    teach = TEACHERS if len(sys.argv) <= 4 else {k: TEACHERS[k] for k in sys.argv[4].split(",")}
    run_all(out, [(r, i, MAKE[r], teach, seeds, {"save_traj": "all"}) for i in range(funcs) for r in MAKE], **kw)
