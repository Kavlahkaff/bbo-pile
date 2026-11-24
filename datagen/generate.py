"""Generate the trajectory corpus.
    python -m datagen.generate --out CORPUS --train 50000 [--task k --n_tasks n] [--workers w]
Function indices per family: BASE + [0, 500) test, [500, 1000) val, [1000, 1000 + train) train; seeds 5 / 2 / 2. Indices
are disjoint from every function used while choosing the optimizers. Raising --train later only adds functions.
Output CORPUS/traj_NNNNN.jsonl, one line per run: fid, family, group, i, split, seed, opt, n_init, k (x * 999), y, lr.
A task is resumable at function level: an incomplete last function is dropped and recomputed."""
import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

BASE, N_TEST, N_VAL = 100_000, 500, 500
SEEDS = {"test": 5, "val": 2, "train": 2}


def split_of(j):
    return "test" if j < N_TEST else "val" if j < N_TEST + N_VAL else "train"


def run_function(job):
    from datagen.functions import Function
    from datagen.optimizers import OPTIMIZERS, Q, initial_design, run
    family, j, opts = job
    fn, split, lines = Function(family, BASE + j), split_of(j), []
    for seed in range(SEEDS[split]):
        n_init = len(initial_design(fn.i, seed))
        for opt in opts or OPTIMIZERS:
            X = run(opt, fn, seed)
            y = fn(X)
            lines.append(json.dumps({"fid": fn.fid, "family": family, "group": fn.group, "i": fn.i, "split": split,
                                     "seed": seed, "opt": opt, "n_init": n_init,
                                     "k": np.rint(X * (Q - 1)).astype(int).tolist(), "y": y.tolist(),
                                     "lr": np.round(fn.log_regret(y), 5).tolist()}))
    return fn.fid, lines


def expected(split, n_opts=3):
    return SEEDS[split] * n_opts


def repair(path):
    """Drop truncated lines and incomplete functions; returns the finished fids."""
    by = {}
    if path.exists():
        for line in open(path):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            by.setdefault(d["fid"], {})[(d["seed"], d["opt"])] = (line, d["split"])
        ok = {f: v for f, v in by.items() if len(v) == expected(next(iter(v.values()))[1])}
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for v in ok.values():
                fh.writelines(l for l, _ in v.values())
        os.replace(tmp, path)
        return set(ok)
    return set()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=50_000, help="train functions per family")
    ap.add_argument("--test_val_only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N jobs of this task")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n_tasks", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    from datagen.functions import FAMILIES
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = N_TEST + N_VAL + (0 if a.test_val_only else a.train)
    jobs = [(fam, j, None) for j in range(n) for fam in FAMILIES][a.task::a.n_tasks][:a.limit]
    path = out / f"traj_{a.task:05d}.jsonl"
    done = repair(path)
    jobs = [x for x in jobs if f"{x[0]}-{BASE + x[1]}" not in done]
    print(f"task {a.task}/{a.n_tasks}: {len(jobs)} functions to run, {len(done)} done", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool, open(path, "a") as fh:
        for m, (_, lines) in enumerate(pool.imap_unordered(run_function, jobs, chunksize=4), 1):
            fh.write("\n".join(lines) + "\n")
            fh.flush()
            if m % max(1, len(jobs) // 20) == 0:
                print(f"  {m}/{len(jobs)} functions, {(time.time() - t0) / 60:.1f} min", flush=True)
    print("task complete", flush=True)


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")                             # inherited by the spawned workers
    main()
