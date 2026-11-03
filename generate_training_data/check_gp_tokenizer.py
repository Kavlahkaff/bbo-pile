"""
Checks for the GP tokenizer (gp_v1):
    - round trip encode/decode on --n lines of every export
    - every number 0..999 and every tag of the symbol file is exactly one token
    - the fixed search-space header always tokenizes the same way
    - tokens per trajectory (mean / p95 / max) per export, split full vs shorter trajectories
    - total train tokens per export (extrapolated from a sample if the export is large)

    python check_gp_tokenizer.py --tokenizer_dir tok/gp_v1 --symbols gp_tokenizer_symbols.txt --exports exp/R exp/O
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

import sentencepiece as spm  # noqa: E402


class Tokenizer:
    """Minimal stand-in for litgpt.Tokenizer on a bare sentencepiece model (no tokenizer_config.json, so litgpt adds
    no BOS either). Avoids importing litgpt -> lightning -> torchmetrics -> torch(vision) for a CPU-only check."""

    def __init__(self, tokenizer_dir):
        self.processor = spm.SentencePieceProcessor(model_file=str(Path(tokenizer_dir) / "tokenizer.model"))
        self.vocab_size = self.processor.vocab_size()

    def encode(self, s, bos=False):
        ids = self.processor.encode(s)
        if bos:
            ids = [self.processor.bos_id()] + ids
        return np.asarray(ids)

    def decode(self, ids):
        return self.processor.decode(list(map(int, ids)))

HEADER = "search-space:{type:UNI,min_value:0.0,max_value:1.0,linear_scale}{type:UNI,min_value:0.0,max_value:1.0,linear_scale}"


def n_trials(line):
    return line.count("|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--exports", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=100, help="lines per export for the round trip")
    ap.add_argument("--sample", type=int, default=20000, help="lines per export for the length statistics")
    ap.add_argument("--max_seq_length", type=int, default=1024)
    args = ap.parse_args()

    tok = Tokenizer(args.tokenizer_dir)
    report, ok = {"vocab_size": tok.vocab_size}, True

    def single(s):
        ids = tok.encode(s, bos=False).tolist()
        # sentencepiece may emit the word-boundary piece first; a symbol counts as one token if it is one piece
        ids = [i for i in ids if tok.processor.id_to_piece(i) != "▁"]
        return len(ids) == 1

    bad_num = [i for i in range(1000) if not single(str(i))]
    symbols = [l.strip() for l in open(args.symbols) if l.strip() and not l.startswith("#")]
    bad_sym = [s for s in symbols if not single(s)]
    report["numbers_not_single_token"] = bad_num
    report["symbols_not_single_token"] = bad_sym
    ok &= not bad_num and not bad_sym

    rnd = random.Random(0)
    headers = set()
    for d in args.exports:
        lines = open(Path(d) / "train.txt", encoding="utf-8").read().splitlines()
        if not lines:
            report[str(d)] = {"lines": 0}
            continue
        sample = rnd.sample(lines, min(args.sample, len(lines)))
        rt_fail = 0
        for line in sample[: args.n]:
            if tok.decode(tok.encode(line, bos=False)) != line:
                rt_fail += 1
        lens = {"full": [], "shorter": []}
        for line in sample:
            n = len(tok.encode(line, bos=False))
            lens["full" if n_trials(line) >= 50 else "shorter"].append(n)
            if line.startswith(HEADER):
                head = line.split(",algorithm:")[0]
                headers.add(tuple(tok.encode(head, bos=False).tolist()))
        all_lens = lens["full"] + lens["shorter"]
        stats = {k: {"n": len(v), "mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)), "max": int(max(v))}
                 for k, v in lens.items() if v}
        total = float(np.mean(all_lens)) * len(lines)
        report[str(d)] = {"lines": len(lines), "roundtrip_failures": rt_fail, "tokens_per_line": stats,
                          "train_tokens_est": int(total)}
        ok &= rt_fail == 0 and max(all_lens) <= args.max_seq_length
    report["distinct_header_tokenizations"] = len(headers)
    ok &= len(headers) == 1
    report["ok"] = bool(ok)
    print(json.dumps(report, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
