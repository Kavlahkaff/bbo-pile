"""Export corpus trajectories to text lines (train.txt, valid.txt). Test functions are never exported.
    python generate_training_data/compile_traces.py --corpus CORPUS --output_path exp/log [--marker]
One line per trajectory: search-space:{...}{...},algorithm:<tag>,history:x0,x1*y|...
Lines: every optimizer under its own tag (GP, NM, RS), and a copy of the family's winner (GP on gp, NM on the others)
under BEST: the family must be inferred in context. --marker writes the end of the initial design into every line."""
import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datagen.functions import HELD_OUT  # noqa: E402
from datagen.optimizers import WINNER  # noqa: E402
from evaluation.prompt import make_history  # noqa: E402

FILES = {"train": "train", "val": "valid"}


def tags(t):
    return [t["opt"]] + (["BEST"] if t["opt"] == WINNER[t["family"]] else [])


def lines_of(t, a):
    return [make_history(tag, t["k"], t["y"], t["n_init"] if a.marker else None).get_prompt() for tag in tags(t)]


def process(job):
    path, part, a = job
    n = Counter()
    with open(f"{part}.train", "w") as tr, open(f"{part}.val", "w") as va:
        for line in open(path):
            t = json.loads(line)
            if t["split"] not in FILES:
                continue
            assert t["family"] not in HELD_OUT, "held-out family in train / val"
            for l in lines_of(t, a):
                (tr if t["split"] == "train" else va).write(l + "\n")
                n[f"{t['split']}_{t['family']}_{l.split('algorithm:')[1].split(',')[0]}"] += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--marker", action="store_true", help="mark the end of the initial design")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    a = ap.parse_args(argv)
    out = Path(a.output_path)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.corpus).glob("traj_*.jsonl"))
    if not files:
        sys.exit(f"no traj_*.jsonl in {a.corpus}")
    parts = [out / f"_part{n:05d}" for n in range(len(files))]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        counts = sum(pool.map(process, [(f, p, a) for f, p in zip(files, parts)], chunksize=1), Counter())
    rng = np.random.default_rng(a.seed)
    for split, name in FILES.items():                              # shuffled in memory (about 300 bytes per line)
        lines = []
        for p in parts:
            lines += open(f"{p}.{split}").read().splitlines()
            Path(f"{p}.{split}").unlink()
        order = rng.permutation(len(lines))
        with open(out / f"{name}.txt", "w") as fh:
            fh.writelines(lines[j] + "\n" for j in order)
    json.dump({"args": vars(a), "counts": dict(sorted(counts.items()))}, open(out / "manifest.json", "w"), indent=1)
    print(json.dumps(dict(sorted(counts.items())), indent=1))


if __name__ == "__main__":
    main()
