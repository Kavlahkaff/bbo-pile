"""
Teacher-forced scoring of test traces (no rollouts): for every test trajectory of every teacher and every prompt tag,
the model's per-step NLL of the teacher's next x (x0, x1) and y (renormalised over the 1000 number tokens), plus the model's greedy next x0 / x1 (restricted to
the 1000 number tokens, each conditioned on the teacher's prefix) and its distance to the teacher's point.

Tags are literal or templates with {K} = function family (see run_gp_benchmark.py). Default: all 9 kernel_mode tags +
UNK, which gives the G1 confusion matrix (rows = teacher, columns = tag) and, via cumulative NLL over steps, the G2
in-context tag posterior.

    python benchmarks/gp_functions/score_teacher_traces.py --corpus corpus_student_40k --checkpoint_dir ckpt/final \
        --subset test_core --output scores.parquet

Output: one row per (fid, seed, mid, tag, t): nll_x0, nll_x1, nll_x (= sum), nll_y, greedy_x0, greedy_x1,
teacher_x0, teacher_x1 (quantized 0..999), dist (euclidean, in [0,1]^2 units), init (t < n_init).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gp_function import CONFIG_SPACE  # noqa: E402
from open_optformer.gp_corpus import KINDS, iter_trajectories, load_specs, quantize_hp  # noqa: E402
from open_optformer.history import History  # noqa: E402

TOKENS_PER_TRIAL = 6  # x0 , x1 * y |
KERNEL_MODE_TAGS = [f"GP-{k.upper()}-{m}" for k in KINDS for m in ("ORACLE", "MLE")] + ["RS", "UNK"]


def load_model(checkpoint_dir, device):
    from litgpt.config import Config
    from litgpt.model import GPT
    from litgpt.tokenizer import Tokenizer
    checkpoint_dir = Path(checkpoint_dir)
    model = GPT(Config.from_file(checkpoint_dir / "model_config.yaml"))
    sd = torch.load(checkpoint_dir / "lit_model.pth", weights_only=True, map_location="cpu")
    model.load_state_dict(sd.get("model", sd))
    return model.to(device).eval(), Tokenizer(checkpoint_dir)


def resolve(template, spec, hp_suffix):
    tag = template.replace("{K}", spec["kind"].upper())
    if hp_suffix and tag != "UNK":
        l0, l1, o = quantize_hp(spec, 0)
        tag += f",hp:{{ls:{l0},{l1},os:{o}}}"
    return tag


def encode_trace(tok, tag, X, y, space_first, prefix_cache, y_norm="global"):
    """Token ids of the full trace and the index of the first history token. Checks that the history part tokenizes
    as exactly 6 tokens per trial, so positions are known without searching."""
    h = History(name="gp", algorithm=tag, config_space=dict(CONFIG_SPACE), remove_names=True, space_first=space_first,
                y_norm=y_norm)
    for (x0, x1), v in zip(X, y):
        h.add_trial({"x0": float(x0), "x1": float(x1)}, float(v))
    line = h.get_prompt()
    ids = tok.encode(line, bos=False).tolist()
    if tag not in prefix_cache:
        prefix = line[: line.index("history:") + len("history:")]
        prefix_cache[tag] = tok.encode(prefix, bos=False).tolist()
    p = prefix_cache[tag]
    assert ids[: len(p)] == p, "prefix tokenizes differently inside the line"
    assert len(ids) - len(p) == TOKENS_PER_TRIAL * len(y), "history is not 6 tokens per trial"
    return ids, len(p)


@torch.no_grad()
def score_batch(model, batch, number_ids, device):
    """batch: list of (ids, start). Returns per sequence an array (T, 5): nll_x0, nll_x1, nll_y, greedy_x0, greedy_x1
    (greedy as 0..999 value)."""
    L = max(len(ids) for ids, _ in batch)
    inp = torch.zeros((len(batch), L), dtype=torch.long)
    for b, (ids, _) in enumerate(batch):
        inp[b, : len(ids)] = torch.tensor(ids)
    logits = model(inp.to(device)).float()
    num = torch.tensor(number_ids, device=device)
    # distribution restricted to the number tokens (the only valid tokens at x0, x1, y). Unrestricted, the mass the
    # model puts on ending the line after '|' (packed training, shortened trajectories) would inflate nll_x0.
    logp = torch.full_like(logits, float("-inf"))
    logp[..., num] = torch.log_softmax(logits[..., num], -1)
    out = []
    for b, (ids, start) in enumerate(batch):
        T = (len(ids) - start) // TOKENS_PER_TRIAL
        pos = start + TOKENS_PER_TRIAL * torch.arange(T, device=device)  # position of x0 of trial t
        tgt = torch.tensor(ids, device=device)
        # the token at position p is predicted from logits at p - 1
        nll = lambda p: -logp[b, p - 1, tgt[p]]
        g0 = logits[b, pos - 1][:, num].argmax(-1)
        g1 = logits[b, pos + 1][:, num].argmax(-1)
        out.append(torch.stack([nll(pos), nll(pos + 2), nll(pos + 4), g0.float(), g1.float()], 1).cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--tags", nargs="+", default=KERNEL_MODE_TAGS)
    ap.add_argument("--hp_suffix", action="store_true")
    ap.add_argument("--subset", choices=["test_core", "test"], default="test_core")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--max_functions", type=int, default=None)
    ap.add_argument("--no_space_first", dest="space_first", action="store_false")
    ap.add_argument("--y_norm", choices=["global", "causal"], default="global",
                    help="y quantization; must match the training export. With causal, teacher forcing sees exactly "
                         "the prompts of a rollout; with global, early y tokens are scaled with the full trace")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    specs = load_specs(args.corpus)
    fids = sorted(f for f, s in specs.items() if s["split"] == "test" and (args.subset == "test" or s["test_core"]))
    if args.max_functions:
        by_kind = {}
        for f in fids:
            by_kind.setdefault(specs[f]["kind"], []).append(f)
        per = max(1, args.max_functions // len(by_kind))
        fids = [f for v in by_kind.values() for f in v[:per]]
    fids = set(fids)
    traces = [t for t in iter_trajectories(args.corpus, fields=("fid", "seed", "mid", "X", "y", "n_init", "key"),
                                           split="test")
              if t["fid"] in fids and t["seed"] in args.seeds]
    model, tok = load_model(args.checkpoint_dir, args.device)
    number_ids = [tok.token_to_id(str(i)) for i in range(1000)]
    print(f"{len(traces)} traces x {len(args.tags)} tags")

    rows, cache, pending = [], {}, []

    def flush():
        res = score_batch(model, [(ids, st) for ids, st, _ in pending], number_ids, args.device)
        for (_, _, meta), r in zip(pending, res):
            t, tag, Xq = meta
            T = len(r)
            rows.append(pd.DataFrame({
                "fid": t["fid"], "family": specs[t["fid"]]["kind"], "seed": t["seed"], "mid": t["mid"], "tag": tag,
                "t": np.arange(T), "init": np.arange(T) < t["n_init"],
                "nll_x0": r[:, 0], "nll_x1": r[:, 1], "nll_x": r[:, 0] + r[:, 1], "nll_y": r[:, 2],
                "greedy_x0": r[:, 3].astype(int), "greedy_x1": r[:, 4].astype(int),
                "teacher_x0": Xq[:, 0], "teacher_x1": Xq[:, 1],
                "dist": np.hypot(r[:, 3] - Xq[:, 0], r[:, 4] - Xq[:, 1]) / 999,
            }))
        pending.clear()

    for t in traces:
        spec = specs[t["fid"]]
        X = np.asarray(t["X"], float)
        Xq = np.rint(np.clip(X, 0, 1) * 999).astype(int)
        for tmpl in args.tags:
            tag = resolve(tmpl, spec, args.hp_suffix)
            ids, start = encode_trace(tok, tag, X, t["y"], args.space_first, cache, args.y_norm)
            pending.append((ids, start, (t, tmpl, Xq)))
            if len(pending) >= args.batch_size:
                flush()
    if pending:
        flush()
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(args.output, index=False)
    print(f"wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
