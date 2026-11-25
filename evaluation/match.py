"""Does the student sample like the teacher it seems to imitate? Sample-based, no classifier.
    python -m evaluation.match teachers --output match/teachers.jsonl                              (CPU, once)
    python -m evaluation.match student --checkpoint_dir ckpt/final --teachers match/teachers.jsonl --output match/run_0.jsonl
    python -m evaluation.match report --teachers ... --run base=match/run [--run base=match/run_s1] --output report.txt

k counts the proposals after the shared initial design (k = 1 is the first point an optimizer chooses).
run   closed loop: M runs per arm (student tag or teacher, other random numbers each) from the same function and initial
      design. "at": distribution of proposal k; "upto": of proposals 1..k together.
next  same history: next point given the first k - 1 proposals of a teacher run; M draws per teacher, --draws of the student. The teacher's draws are
      from its exact conditional distribution: GP refits on the history; NM replays, and only simplex directions that no
      point has revealed yet are drawn again (so NM exists on NM histories and at k = 1 only).
Space "xy": position in the unit square. "step": log10 distance to the incumbent (>= 1e-3), which still resolves the late,
      local phase. Distance: energy distance 2 E|A-B| - E|A-A'| - E|B-B'|, unbiased (pairs of different runs only):
      0 = same distribution, small negative values are noise.
Report, winner = the family's winning teacher, loser = the other one of GP and NM:
  to_GP/NM/RS  distance student(tag) to the teachers;  own = student tagged with the winner to the winner (what imitation
               on command achieves);  floor = two halves of the winner's runs (must be about 0);  GP_NM = the two teachers
  margin       distance to loser - distance to winner, > 0: samples like the winner
  closeness    1 - distance to winner / distance between the teachers: 1 = same distribution, 0 = as far as the other teacher
  near_win     share of functions whose nearest teacher (of GP, NM, RS) is the family's winner
  near_better  the same for the teacher that is better (mean log regret over its M runs) on that very function
Spread table (a distance mixes "wrong place" and "wrong variance"):
  spread_S / spread_own / spread_T   E|A-A'| of the student, of the student tagged with the winner, of the winning teacher
  shift / shift_own   distance of the two arms' means to the winner's mean (root mean square over functions, noise removed)
Mass table (next only): exact probability, no student sampling, that the next point falls within R grid steps (both
coordinates, 1 step = 0.001) of a draw of the teacher:
  GP, NM  near that teacher's draws;  both  near both;  neither  the rest;  self_*  share of the teacher's draws near one of
  its other draws (what a perfect student gets, roughly);  area_*  the same probability for uniform sampling
Means over functions, confidence intervals from a bootstrap over functions; model seeds are averaged per function first."""
import argparse
import glob
import json
import os
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from datagen.functions import FAMILIES, HELD_OUT, Function, seed_of
from datagen.generate import BASE, N_TEST
from datagen.optimizers import OPTIMIZERS, Q, TRIALS, WINNER, gp_ucb, initial_design, nelder_mead, random_search

PROPOSALS, KS = 40, (1, 2, 3, 5, 10, 20, 40)          # at most 10 initial points, so every run has 40 proposals
SPACES, HISTORIES, RADII = ("xy", "step"), ("GP", "NM"), (2, 10)


def grid(X):
    return np.rint(np.asarray(X) * (Q - 1)).astype(int).tolist()


def lines(path):
    """JSON lines, without one cut by a time limit."""
    for line in open(path):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            pass


def open_resumed(path):
    """Open for append, on a fresh line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cut = path.exists() and path.stat().st_size and not open(path, "rb").read().endswith(b"\n")
    fh = open(path, "a")
    if cut:
        fh.write("\n")
    return fh


# --------------------------------------------------------------------------------------------- teachers
def key_of(fn, seed, rep, opt):
    return f"{fn.fid}|{seed}-rep{rep}|{OPTIMIZERS[opt]}"


def teacher_run(opt, fn, seed, rep):
    return {"GP": gp_ucb, "NM": nelder_mead, "RS": random_search}[opt](key_of(fn, seed, rep, opt), fn, initial_design(fn.i, seed))


def nm_next(fn, seed, rep, X, n, draw):
    """One draw of NM's point n given its own points X[:n] (run `rep`). Simplex edge j of a (re)start at m points is shown
    by point m + 1 + j, so its direction is fixed if that point is in the history."""
    old = np.random.default_rng(seed_of("nm", key_of(fn, seed, rep, "NM").rsplit("|", 1)[0]))
    new = np.random.default_rng(seed_of("nm-next", fn.fid, seed, rep, n, draw))

    def signs(m):
        a, b = old.choice([-1.0, 1.0], 2), new.choice([-1.0, 1.0], 2)
        return np.where(m + 1 + np.arange(2) < n, a, b)

    Z = nelder_mead("", fn, initial_design(fn.i, seed), signs=signs)
    assert (grid(Z[:n]) == grid(X[:n])), "the replay left the history"
    return Z[n]


def histories():
    return [("init", 1)] + [(h, k) for h in HISTORIES for k in KS if k > 1]


def teacher_function(job):
    fam, i, seed, M = job
    fn = Function(fam, i)
    n0 = len(initial_design(i, seed))
    X = {o: [teacher_run(o, fn, seed, m) for m in range(M)] for o in OPTIMIZERS}
    nxt = {}
    for h, k in histories():
        n = n0 + k - 1
        H = X["NM" if h == "init" else h][0]
        g = [gp_ucb(f"next|{fn.fid}|{seed}|{h}|{k}|{m}", fn, H[:n], stop=n + 1)[n] for m in range(M)]
        r = np.random.default_rng(seed_of("rs-next", fn.fid, seed, h, k)).uniform(size=(M, 2))
        d = {"GP": grid(g), "RS": grid(r)}
        if h != "GP":
            d["NM"] = grid([nm_next(fn, seed, 0, H, n, m) for m in range(M)])
        nxt[f"{h}|{k}"] = d
    y = {o: [fn(x).tolist() for x in X[o]] for o in X}
    score = {o: float(np.mean([fn.log_regret(v) for v in y[o]])) for o in y}
    return json.dumps({"fid": fn.fid, "family": fam, "i": i, "seed": seed, "n_init": n0, "M": M, "score": score,
                       "k": {o: [grid(x) for x in X[o]] for o in X}, "y": y, "next": nxt})


def cmd_teachers(a):
    out = Path(a.output)
    done = {}
    if out.exists():
        done = {d["fid"]: d["M"] for d in lines(out)}
        assert set(done.values()) <= {a.reps}, "existing file has another number of runs; use a new --output"
    jobs = [(fam, BASE + j, 0, a.reps) for j in range(a.functions) for fam in FAMILIES if f"{fam}-{BASE + j}" not in done]
    print(f"{len(jobs)} functions to run, {len(done)} done", flush=True)
    t0 = time.time()
    with open_resumed(out) as fh, get_context("spawn").Pool(a.workers) as pool:
        for m, line in enumerate(pool.imap_unordered(teacher_function, jobs), 1):
            fh.write(line + "\n")
            fh.flush()
            if m % 25 == 0:
                print(f"  {m}/{len(jobs)}, {(time.time() - t0) / 60:.1f} min", flush=True)


# --------------------------------------------------------------------------------------------- student
def near(points, r):
    """1000 x 1000 mask: cells within r grid steps of a point."""
    Z = np.zeros((Q, Q), bool)
    for a, b in points:
        Z[max(a - r, 0):a + r + 1, max(b - r, 0):b + r + 1] = True
    return Z


def next_distribution(student, d, tag, draws, batch):
    """Draws of the student's next point and exact masses near the teachers' draws, for every history of one function.
    p(k0) takes one pass per history and p(k1 | k0) one per distinct k0 that is drawn or near a teacher's draw."""
    import torch

    from evaluation.prompt import make_history
    prompts = []
    for h, k in histories():
        o, L = "NM" if h == "init" else h, d["n_init"] + k - 1
        prompts.append(student.tok.encode(make_history(tag, student.y_norm, d["k"][o][0][:L], d["y"][o][0][:L]).get_prompt(),
                                          bos=False).tolist())
    P0 = student.number_probs(prompts, batch)
    pts, mass = {}, {}
    plan, ids = [], []
    for j, (h, k) in enumerate(histories()):
        g = torch.Generator().manual_seed(seed_of("next", d["fid"], tag, h, k) % 2**31)
        k0 = torch.multinomial(P0[j], draws, replacement=True, generator=g).tolist()
        T = {o: v for o, v in d["next"][f"{h}|{k}"].items() if o != "RS"}
        rows = set(k0) | {x for v in T.values() for a, _ in v for x in range(max(a - max(RADII), 0), min(a + max(RADII), Q - 1) + 1)}
        rows = sorted(rows)
        plan.append((h, k, g, k0, T, {x: len(ids) + i for i, x in enumerate(rows)}))
        ids += [prompts[j] + [int(student.numbers[x]), student.comma] for x in rows]
    P1 = student.number_probs(ids, batch)
    for j, (h, k, g, k0, T, at) in enumerate(plan):
        k1 = torch.multinomial(P1[[at[x] for x in k0]], 1, generator=g)[:, 0].tolist()
        pts[f"{h}|{k}"] = [list(p) for p in zip(k0, k1)]
        rows = list(at)
        J = P0[j, rows].double().numpy()[:, None] * P1[[at[x] for x in rows]].double().numpy()
        m = {}
        for r in RADII:
            Z = {o: near(v, r)[rows] for o, v in T.items()}
            if len(Z) == 2:
                Z["both"] = Z["GP"] & Z["NM"]
            m.update({f"{o}|{r}": float(J[z].sum()) for o, z in Z.items()})
        mass[f"{h}|{k}"] = m
    return {"next": pts, "mass": mass}


def cmd_student(a):
    import torch

    from evaluation.rollout import Student
    T = sorted(lines(a.teachers), key=lambda d: (d["i"], d["family"]))
    s, n = map(int, a.shard.split("/"))
    T = T[s::n]
    out = Path(a.output)
    done = {(d["fid"], d["tag"], d["what"]) for d in lines(out)} if out.exists() else set()
    student = Student(a.checkpoint_dir, a.y_norm, a.temperature)
    t0 = time.time()
    with open_resumed(out) as fh:
        for c, d in enumerate(T, 1):
            fn, M, seed = Function(d["family"], d["i"]), d["M"], d["seed"]

            def save(tag, what, **kw):
                fh.write(json.dumps({"fid": d["fid"], "tag": tag, "what": what, **kw}) + "\n")

            todo = [t for t in a.tags if (d["fid"], t, "run") not in done]
            J = [(t, m) for t in todo for m in range(M)]
            R = []
            for b in range(0, len(J), a.batch):
                R += student.run_batch([(fn, seed, t) for t, _ in J[b:b + a.batch]], [m for _, m in J[b:b + a.batch]])
            for x, t in enumerate(todo):
                save(t, "run", k=[r[0] for r in R[x * M:(x + 1) * M]], y=[r[1] for r in R[x * M:(x + 1) * M]])
            for t in [t for t in a.tags if (d["fid"], t, "next") not in done]:
                save(t, "next", **next_distribution(student, d, t, a.draws, a.batch))
            fh.flush()
            print(f"  {c}/{len(T)}, {(time.time() - t0) / 60:.1f} min", flush=True)


# --------------------------------------------------------------------------------------------- distances
def log_step(x, inc):
    return np.log10(np.maximum(np.linalg.norm(x - inc, axis=-1), 1e-3))[..., None]


def features(k, y, n0):
    """Runs (M x TRIALS) -> space -> M x PROPOSALS x dim."""
    X, y = np.asarray(k) / (Q - 1), np.asarray(y)
    best = np.array([[int(v[:t].argmin()) for t in range(n0, n0 + PROPOSALS)] for v in y])
    P = X[:, n0:n0 + PROPOSALS]
    return {"xy": P, "step": log_step(P, np.take_along_axis(X, best[..., None], 1))}


def mean_dist(A, B, same, power=1):
    """Mean (distance ** power) per pair of proposals (T x T), over pairs of runs (M x T x dim each); same: without a run with itself."""
    m, T = len(A), A.shape[1]
    D = np.linalg.norm(A.reshape(m * T, 1, -1) - B.reshape(1, len(B) * T, -1), axis=-1).reshape(m, T, len(B), T) ** power
    if not same:
        return D.mean((0, 2))
    return (D.sum((0, 2)) - np.einsum("itis->ts", D)) / (m * (m - 1))


class Distances:
    """Energy distances between named arms (M x T x dim)."""

    def __init__(self, arms):
        self.arms, self.c = arms, {}

    def _mean(self, a, b, power=1):
        if (a, b, power) not in self.c:
            self.c[a, b, power] = mean_dist(self.arms[a], self.arms[b], a == b, power)
        return self.c[a, b, power]

    @staticmethod
    def _both(C):
        """(at k, over proposals 1..k) of a matrix over pairs of proposals."""
        return np.diag(C), np.array([C[:k, :k].mean() for k in range(1, len(C) + 1)])

    def __call__(self, a, b):
        return self._both(2 * self._mean(a, b) - self._mean(a, a) - self._mean(b, b))

    def spread(self, a):
        return self._both(self._mean(a, a))

    def shift2(self, a, b):
        """Squared distance of the means, unbiased."""
        return self._both(self._mean(a, b, 2) - (self._mean(a, a, 2) + self._mean(b, b, 2)) / 2)

    def all(self, tags):
        """(name, (at, upto)) of everything in the report."""
        out = [(n, self(a, b)) for n, a, b in pairs(self.arms, tags)]
        out += [(f"spread:{n}", self.spread(n)) for n in self.arms if n[0] in "ST"]
        return out + [(f"shift2:{n}", self.shift2(a, b)) for n, a, b in pairs(self.arms, tags) if a[0] == "S"]


def pairs(arms, tags):
    """(name, a, b) of the distances in the report; teachers are T:<opt>, students S:<tag>."""
    out = [(f"S:{t}>{o}", f"S:{t}", f"T:{o}") for t in tags for o in OPTIMIZERS]
    out += [(f"T:{a}>{b}", f"T:{a}", f"T:{b}") for a, b in (("GP", "NM"), ("GP", "RS"), ("NM", "RS"))]
    out += [(f"half:{o}", f"H0:{o}", f"H1:{o}") for o in OPTIMIZERS]
    return [p for p in out if p[1] in arms and p[2] in arms]


def with_halves(arms):
    h = {f"H{j}:{n[2:]}": v[j::2] for n, v in arms.items() if n.startswith("T:") for j in (0, 1)}
    return {**arms, **h}


def function_rows(job):
    """All distances of one function and one model: [(test, k, space, name, value)]."""
    d, S = job
    rows, tags = [], sorted({t for t, _ in S})
    F = {f"T:{o}": features(d["k"][o], d["y"][o], d["n_init"]) for o in OPTIMIZERS}
    F.update({f"S:{t}": features(S[t, "run"]["k"], S[t, "run"]["y"], d["n_init"]) for t in tags})
    for sp in SPACES:
        D = Distances(with_halves({n: f[sp] for n, f in F.items()}))
        for name, (at, upto) in D.all(tags):
            for k in KS:
                rows += [("run-at", k, sp, name, at[k - 1]), ("run-upto", k, sp, name, upto[k - 1])]
    for h, k in histories():
        o, L = "NM" if h == "init" else h, d["n_init"] + k - 1
        P = {f"T:{q}": v for q, v in d["next"][f"{h}|{k}"].items()}
        P.update({f"S:{t}": S[t, "next"]["next"][f"{h}|{k}"] for t in tags})
        P = {q: np.asarray(v) / (Q - 1) for q, v in P.items()}
        inc = np.array(d["k"][o][0][int(np.argmin(d["y"][o][0][:L]))]) / (Q - 1)
        for sp in SPACES:
            D = Distances(with_halves({q: (v if sp == "xy" else log_step(v, inc))[:, None] for q, v in P.items()}))
            rows += [(f"next-{h}", k, sp, name, v[0][0]) for name, v in D.all(tags)]
        for t in tags:
            rows += [(f"next-{h}", k, "xy", f"mass:S:{t}:{n}", v) for n, v in S[t, "next"]["mass"][f"{h}|{k}"].items()]
        for q in ("GP", "NM"):
            v = np.array(d["next"][f"{h}|{k}"].get(q, []))
            for r in RADII if len(v) else ():
                close = np.abs(v[:, None] - v[None]).max(-1) <= r
                np.fill_diagonal(close, False)
                rows += [(f"next-{h}", k, "xy", f"mass:self:{q}|{r}", close.any(1).mean()),
                         (f"next-{h}", k, "xy", f"mass:area:{q}|{r}", near(v, r).mean())]
    return rows


# --------------------------------------------------------------------------------------------- report
def load_student(prefix):
    S = defaultdict(dict)
    for p in sorted(glob.glob(f"{prefix}_[0-9]*.jsonl")):
        for d in lines(p):
            S[d["fid"]][d["tag"], d["what"]] = d
    tags = {t for v in S.values() for t, _ in v}
    return {f: v for f, v in S.items() if len(v) == 2 * len(tags)}, tags


def summarise(V, meta, tag):
    """V: (model, fid) -> {name: value} of one test, k, space and family. Returns one report row."""
    from datagen.check_corpus import boot_ci
    fids = sorted({f for _, f in V})
    win = WINNER[meta[fids[0]]["family"]]
    lose = "NM" if win == "GP" else "GP"

    by = defaultdict(list)
    for (_, f), v in V.items():
        by[f].append(v)

    def per_fn(g):
        """Mean over the models per function."""
        return np.array([np.mean([g(v) for v in by[f]]) for f in fids], float)

    def get(name):
        return per_fn(lambda v: v[name]) if name in next(iter(V.values())) else None

    def nearest(target):
        def g(v, t):
            c = {o: v[f"S:{tag}>{o}"] for o in OPTIMIZERS}
            return min(c, key=c.get) == t
        return np.mean([np.mean([g(v, target(f)) for v in by[f]]) for f in fids])

    to = {o: get(f"S:{tag}>{o}") for o in OPTIMIZERS}
    gn, own, half = get("T:GP>NM"), get(f"S:{win}>{win}"), get(f"half:{win}")
    r = {**{f"to_{o}": x for o, x in to.items()}, "own": own, "floor": half, "GP_NM": gn}
    r = {c: "-" if x is None else f"{x.mean():.4f}" for c, x in r.items()}
    r.update(margin="-", closeness="-", near_win="-", near_better="-")
    ci = lambda x: "{:+.4f} [{:+.4f}, {:+.4f}]".format(x.mean(), *boot_ci(x))
    if to["GP"] is not None and to["NM"] is not None:
        r["margin"] = ci(to[lose] - to[win])
        idx = np.random.default_rng(0).integers(0, len(fids), (2000, len(fids)))
        b = np.array([to[win][j].mean() / gn[j].mean() if gn[j].mean() > 0 else np.nan for j in idx])
        if gn.mean() > 0 and np.isfinite(b).all():
            lo, hi = np.percentile(1 - b, [2.5, 97.5])
            r["closeness"] = f"{1 - to[win].mean() / gn.mean():.2f} [{lo:.2f}, {hi:.2f}]"
        r["near_win"] = f"{nearest(lambda f: win):.2f}"
        r["near_better"] = f"{nearest(lambda f: meta[f]['better']):.2f}"
    rms = lambda n: None if get(n) is None else np.sqrt(max(get(n).mean(), 0))
    mean = lambda n: None if get(n) is None else get(n).mean()
    num = lambda x: "-" if x is None else f"{x:.4f}"
    spread = {"spread_S": num(mean(f"spread:S:{tag}")), "spread_own": num(mean(f"spread:S:{win}")),
              "spread_T": num(mean(f"spread:T:{win}")), "shift": num(rms(f"shift2:S:{tag}>{win}")),
              "shift_own": num(rms(f"shift2:S:{win}>{win}"))}
    mass = {}
    for q in RADII:
        m = {o: mean(f"mass:S:{tag}:{o}|{q}") for o in ("GP", "NM", "both")}
        if m["GP"] is None:
            continue
        rest = 1 - m["GP"] - (m["NM"] or 0) + (m["both"] or 0)
        x = {"R": q, **m, "neither": rest, **{f"{w}_{o}": mean(f"mass:{w}:{o}|{q}") for w in ("self", "area") for o in ("GP", "NM")}}
        mass[q] = {c: v if c == "R" else num(v) for c, v in x.items()}
    return r, spread, mass


def cmd_report(a):
    import pandas as pd
    meta = {}
    for d in lines(a.teachers):
        s = d["score"]
        meta[d["fid"]] = {**d, "better": "GP" if s["GP"] < s["NM"] else "NM"}
    runs = defaultdict(list)
    for r in a.run:
        v, prefix = r.split("=", 1)
        runs[v].append(prefix)
    out, raw = [f"{len(meta)} test functions, {next(iter(meta.values()))['M']} runs or draws per arm; definitions: evaluation/match.py"], []
    for fam in FAMILIES:
        m = [x for x in meta.values() if x["family"] == fam]
        if m:
            out.append(f"  {fam:10} winner {WINNER[fam]}: better on {np.mean([x['better'] == WINNER[fam] for x in m]):.0%} "
                       f"of the functions{', HELD OUT' if fam in HELD_OUT else ''}")
    pool = get_context("spawn").Pool(a.workers)
    for v, prefixes in runs.items():
        cells, n_models, all_tags = defaultdict(dict), 0, set()
        for model, prefix in enumerate(prefixes):
            S, tags = load_student(prefix)
            fids = sorted(set(S) & set(meta))
            if not fids:
                out.append(f"MISSING {v} model {model}: no results at {prefix}")
                continue
            if len(fids) < len(meta):
                out.append(f"WARNING {v} model {model}: {len(fids)} of {len(meta)} functions are complete")
            n_models += 1
            all_tags |= tags
            for fid, rows in zip(fids, pool.map(function_rows, [(meta[f], S[f]) for f in fids])):
                for test, k, sp, name, val in rows:
                    cells[meta[fid]["family"], test, sp, k].setdefault((model, fid), {})[name] = val
                    raw.append((v, model, meta[fid]["family"], fid, meta[fid]["better"], test, sp, k, name, val))
        if not n_models:
            continue
        # a function must be complete in every model, else the models are not averaged over the same functions
        for key, V in cells.items():
            cnt = defaultdict(int)
            for _, f in V:
                cnt[f] += 1
            cells[key] = {q: x for q, x in V.items() if cnt[q[1]] == n_models}
        for tag in [t for t in a.tags if t in all_tags]:
            tab = defaultdict(list)
            for (f, t, sp, k), V in cells.items():
                if V:
                    dist, spread, mass = summarise(V, meta, tag)
                    key = {"family": f, "k": k}
                    tab[sp, t, "distance"].append({**key, **dist})
                    tab[sp, t, "spread"].append({**key, **spread})
                    tab[sp, t, "mass"] += [{**key, **m} for m in mass.values()]
            for (sp, t, what), rows in sorted(tab.items(), key=lambda x: (SPACES.index(x[0][0]), x[0][1:])):
                if rows:
                    Z = pd.DataFrame(rows)
                    out.append(f"\n== {v}, tag {tag}, {t}, space {sp}, {what}, {n_models} model seed(s)")
                    out.append(Z.sort_values(["family", "k"] + (["R"] if "R" in Z else [])).to_string(index=False))
    pool.close()
    text = "\n".join(out)
    print(text)
    if a.output:
        Path(a.output).write_text(text + "\n")
        pd.DataFrame(raw, columns=["variant", "model", "family", "fid", "better", "test", "space", "k", "distance", "value"]) \
            .to_csv(Path(a.output).with_suffix(".csv.gz"), index=False, float_format="%.6g")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("teachers")
    p.add_argument("--functions", type=int, default=100, help="test functions per family")
    p.add_argument("--reps", type=int, default=16, help="runs / draws per arm (even)")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--output", required=True)
    p = sub.add_parser("student")
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--teachers", required=True)
    p.add_argument("--tags", nargs="+", default=["GP", "NM", "BEST"])
    p.add_argument("--y_norm", default="causal_log", help="must match the export")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--draws", type=int, default=128, help="student draws of the next point per history")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--output", required=True)
    p = sub.add_parser("report")
    p.add_argument("--teachers", required=True)
    p.add_argument("--run", action="append", required=True, help="variant=prefix (files prefix_<shard>.jsonl), once per model seed")
    p.add_argument("--tags", nargs="+", default=["BEST", "UNK", "GP", "NM"], help="tags to report")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--output", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "teachers":
        assert a.functions <= N_TEST and a.reps >= 4 and a.reps % 2 == 0
    assert a.cmd != "student" or a.temperature > 0, "a distribution needs a temperature"
    {"teachers": cmd_teachers, "student": cmd_student, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    main()
