"""
Completeness check before analyze: rollouts per label and tag, score files, final checkpoints.

    python benchmarks/gp_functions/check_gp_runs.py --out $OUT --base $BASE \
        --labels R O R_T0.5 R_greedy O_T0.5 O_greedy R_1.2B R_10M_1.2B O_none O_random O_causal \
        --scores R R_1.2B R_10M_1.2B

A label is complete when every tag it was run with has --runs_per_tag distinct (fid, seed) runs (symmetry 0;
default 400 test_core functions x 5 seeds) and its shard files have no cut-off lines. Exit code 1 if anything is
missing, so it can gate the analyze job.
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path


def check_label(out, label, expect):
    files = sorted(glob.glob(f"{out}/rollouts/{label}_shard_*.jsonl"))
    if not files:
        return False, f"{label:14s} MISSING  no rollout files"
    runs, bad, temps = defaultdict(set), 0, set()
    for p in files:
        for line in open(p):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if d.get("symmetry", 0) == 0:
                runs[d["tag"]].add((d["fid"], d["seed"]))
                temps.add(d.get("temperature"))
    shards = {Path(p).name.rsplit("_shard_", 1)[1] for p in files}
    n_total = {s.split("of")[1].split(".")[0] for s in shards}
    per_tag = ", ".join(f"{t}={len(v)}" for t, v in sorted(runs.items()))
    ok = bool(runs) and all(len(v) >= expect for v in runs.values())
    shard_ok = len(n_total) == 1 and len(shards) == int(next(iter(n_total)))
    msg = (f"{label:14s} {'OK      ' if ok and shard_ok else 'INCOMPLETE'}  {len(files)} shards"
           f"{'' if shard_ok else ' (shard set incomplete or mixed: ' + ' '.join(sorted(shards)) + ')'}, "
           f"T={sorted(temps, key=str)}, {per_tag}" + (f", {bad} cut-off lines" if bad else ""))
    return ok and shard_ok, msg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--scores", nargs="*", default=[])
    ap.add_argument("--runs_per_tag", type=int, default=2000)
    args = ap.parse_args()
    all_ok = True

    print("== rollouts (distinct (fid, seed) runs per tag, symmetry 0)")
    for label in args.labels:
        ok, msg = check_label(args.out, label, args.runs_per_tag)
        all_ok &= ok
        print("  " + msg)

    print("== scores")
    for label in args.scores:
        p = Path(args.out) / f"scores_{label}.parquet"
        if not p.exists():
            all_ok = False
            print(f"  {label:14s} MISSING  {p}")
            continue
        info = f"{p.stat().st_size / 1e6:.1f} MB"
        try:
            import pandas as pd
            s = pd.read_parquet(p)
            info += f", {len(s)} rows" + (f", {s.tag.nunique()} tags" if "tag" in s else "")
        except ImportError:
            pass
        except Exception as e:  # unreadable = cut off
            all_ok = False
            print(f"  {label:14s} BROKEN   {e}")
            continue
        print(f"  {label:14s} OK       {info}")

    print("== checkpoints (final/)")
    for c in sorted(glob.glob(f"{args.base}/checkpoints/*/gonogo_*")):
        final = Path(c) / "final"
        has = final.is_dir() and any(final.glob("*.pth"))
        print(f"  {'OK      ' if has else 'NO FINAL'}  {Path(c).name}")

    print("\nALL COMPLETE - start analyze" if all_ok else "\nNOT COMPLETE - see above")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
