"""Variants against a baseline over several model seeds.
    python -m evaluation.compare --base log --run log=r_2.4B --run log_marker=... [--run log=r_2.4B_s1 ...] --output f
Each --run is variant=rollout-prefix (files prefix_<shard>.jsonl), one per model seed; seeds are matched by the order given.
Score: mean_lr per family and tag. "spread" is the standard deviation between model seeds. Differences are paired by
model seed (negative = variant better than base); variants with one seed get no spread, so treat those as single runs."""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from evaluation.analyze import row


def scores(prefix):
    runs = {}
    for p in sorted(glob.glob(f"{prefix}_[0-9]*.jsonl")):
        for line in open(p):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            runs[d["fid"], d["seed"], d["tag"]] = d
    if not runs:
        return None
    return pd.DataFrame([row(d, d["tag"]) for d in runs.values()]).groupby(["family", "who"]).mean_lr.agg(["mean", "size"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="variant=rollout prefix")
    ap.add_argument("--base", default="log")
    ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)
    by, out = defaultdict(dict), []
    n_seen = defaultdict(int)
    for r in a.run:
        v, prefix = r.split("=", 1)
        s, n_seen[v] = n_seen[v], n_seen[v] + 1
        t = scores(prefix)
        if t is None:
            out.append(f"MISSING {v} seed {s}: no rollouts at {prefix}")
        else:
            by[v][s] = t
    sizes = {int(t["size"].max()) for d in by.values() for t in d.values()}
    if len(sizes) > 1 or any(t["size"].nunique() > 1 for d in by.values() for t in d.values()):
        out.append("WARNING: rollouts are incomplete, the runs differ in size; do not compare yet")
    rows = []
    for v, d in by.items():
        for key in sorted(set().union(*(t.index for t in d.values()))):
            x = {s: t["mean"][key] for s, t in d.items() if key in t.index}
            b = {s: t["mean"][key] for s, t in by.get(a.base, {}).items() if key in t.index}
            both = [x[s] - b[s] for s in x if s in b] if v != a.base else []
            rows.append({"family": key[0], "tag": key[1], "variant": v, "seeds": len(x), "mean_lr": np.mean(list(x.values())),
                         "spread": np.std(list(x.values()), ddof=1) if len(x) > 1 else np.nan,
                         "vs_base": np.mean(both) if both else np.nan,
                         "vs_base_spread": np.std(both, ddof=1) if len(both) > 1 else np.nan, "pairs": len(both)})
    T = pd.DataFrame(rows)
    if len(T):
        out.append(T.sort_values(["family", "tag", "variant"]).round(3).to_string(index=False))
    text = "\n".join(out)
    print(text)
    if a.output:
        open(a.output, "w").write(text + "\n")


if __name__ == "__main__":
    main()
