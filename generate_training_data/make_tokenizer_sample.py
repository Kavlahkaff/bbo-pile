"""
Mixed tokenizer training sample: about --n_lines lines drawn evenly from several exports (one per tag scheme,
with hp suffix, symmetries and shorter trajectories), so every tag is covered.

    python make_tokenizer_sample.py --exports exp/R exp/kernel exp/generic exp/quality exp/kernel_hp ... \
        --n_lines 50000 --output tok_sample.txt
"""
import argparse
import random
from pathlib import Path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", nargs="+", required=True, help="export dirs (train.txt + valid.txt each)")
    ap.add_argument("--n_lines", type=int, default=50_000)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    per = args.n_lines // len(args.exports)
    out = []
    for d in args.exports:
        # reservoir sampling: exports can be larger than memory
        res, n = [], 0
        for name in ("train.txt", "valid.txt"):
            p = Path(d) / name
            if not p.exists():
                continue
            for line in open(p, encoding="utf-8"):
                n += 1
                if len(res) < per:
                    res.append(line)
                else:
                    j = rnd.randrange(n)
                    if j < per:
                        res[j] = line
        print(f"{d}: {len(res)} of {n} lines")
        out += res
    rnd.shuffle(out)
    Path(args.output).write_text("".join(out), encoding="utf-8")
    print(f"wrote {len(out)} lines to {args.output}")
