"""
Closed-loop rollouts of a trained student on GP test functions, from the teachers' initial designs.

For every (function, seed, tag[, symmetry]) the student starts from the teacher's initial design X[:n_init] of that
(function, seed) (mapped through the symmetry), then proposes the remaining points itself. The model is loaded once.

Tags are templates: {K} is replaced by the function family (RBF, M12, M32, M52), so 'GP-{K}-ORACLE' is the matched
oracle tag. --hp_suffix appends the true hyperparameters (as the exporter does). 'RS' as method runs syne-tune random
search instead of the student (sanity baseline; the teacher RS traces are in the corpus anyway).

    python benchmarks/gp_functions/run_gp_benchmark.py --corpus corpus_student_40k --checkpoint_dir ckpt/final \
        --tags 'GP-{K}-ORACLE' 'GP-{K}-MLE' RS UNK --subset test_core --seeds 0 1 2 3 4 --output rollouts.jsonl

Output: one JSON line per run {fid, family, seed, tag, prompt_tag, symmetry, n_init, X, y, log_regret, n_invalid
(always 0: sampling is restricted to number tokens), p_nonnumber, boundary, edge_dist}. Existing (fid, seed, tag, symmetry) entries in --output are skipped, so runs can be resumed or
split across jobs with --shard i/n.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generate_training_data"))

from gp_function import CONFIG_SPACE, GPGridFunction  # noqa: E402
from open_optformer.gp_corpus import (apply_sym, edge_distance, iter_trajectories, load_specs,  # noqa: E402
                                      quantize_hp, seed_of)
from open_optformer.history import History  # noqa: E402

logger = logging.getLogger(__name__)


def resolve_tag(template: str, spec: dict, symmetry: int, hp_suffix: bool) -> str:
    tag = template.replace("{K}", spec["kind"].upper())
    if hp_suffix and tag != "UNK":
        l0, l1, o = quantize_hp(spec, symmetry)
        tag += f",hp:{{ls:{l0},{l1},os:{o}}}"
    return tag


def initial_designs(corpus_dirs, fids, seeds):
    """(fid, seed) -> initial design X[:n_init], taken from the RS teacher trace (all teachers share it)."""
    out = {}
    for t in iter_trajectories(corpus_dirs, fields=("key", "fid", "seed", "mid", "X", "n_init")):
        if t["mid"] == "rs" and t["fid"] in fids and t["seed"] in seeds:
            out[(t["fid"], t["seed"])] = np.asarray(t["X"][: t["n_init"]], float)
    return out


class Student:
    """The trained model as an optimizer, loaded once and reused for many runs.

    Each new point is sampled token by token (temperature 1) with the next-token distribution restricted to the 1000
    number tokens: x0 ~ p(. | prompt), then x1 ~ p(. | prompt, x0, ','). The restriction conditions on "the
    trajectory continues with a valid point". It is needed because training packs lines without a separator, and R
    contains shortened trajectories (1/5/10/20 trials), so after '|' the model also puts mass on the start of a new
    line ('▁search-space...'). That mass is recorded per run as p_nonnumber (mean over the proposed points)."""

    def __init__(self, checkpoint_dir, space_first=True, device=None):
        from score_teacher_traces import load_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tok = load_model(checkpoint_dir, self.device)
        self.number_ids = torch.tensor([self.tok.token_to_id(str(i)) for i in range(1000)], device=self.device)
        self.comma_id = self.tok.token_to_id(",")
        self.max_len = self.model.max_seq_length
        self.space_first = space_first

    @torch.no_grad()
    def _next_logits(self, ids):
        x = torch.tensor(ids[-self.max_len:], device=self.device)[None]
        return self.model(x)[0, -1].float()

    def _sample_number(self, ids, gen):
        logits = self._next_logits(ids)
        p_all = torch.softmax(logits, -1)
        p_num = p_all[self.number_ids]
        mass = float(p_num.sum())
        k = int(torch.multinomial(p_num / p_num.sum(), 1, generator=gen))
        return k, 1.0 - mass

    def run(self, f: GPGridFunction, X0: np.ndarray, tag: str, n_trials: int, seed: int):
        gen = torch.Generator(device=self.device).manual_seed(seed)
        h = History(name="gp", algorithm=tag, config_space=dict(CONFIG_SPACE), remove_names=True,
                    space_first=self.space_first)
        X, y, nonnum = [], [], []
        for i in range(n_trials):
            if i < len(X0):
                x = [float(X0[i, 0]), float(X0[i, 1])]
            else:
                ids = self.tok.encode(h.get_prompt(), bos=False).tolist()
                k0, m0 = self._sample_number(ids, gen)
                k1, m1 = self._sample_number(ids + [int(self.number_ids[k0]), self.comma_id], gen)
                x = [k0 / 999, k1 / 999]  # = history.dequantize on [0, 1]
                nonnum += [m0, m1]
            v = float(f({"x0": x[0], "x1": x[1]})["y"])
            h.add_trial({"x0": x[0], "x1": x[1]}, v)
            X.append(x)
            y.append(v)
        return np.asarray(X), np.asarray(y), float(np.mean(nonnum)) if nonnum else 0.0


def random_search(f, X0, n_trials, seed):
    r = np.random.default_rng(seed)
    X = np.vstack([X0, r.uniform(size=(n_trials - len(X0), 2))])
    return X, f.evaluate(X), 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--checkpoint_dir", default=None, help="litgpt checkpoint dir (lit_model.pth, model_config.yaml, "
                                                            "tokenizer.model); not needed for --tags RS-baseline")
    ap.add_argument("--tags", nargs="+", default=["GP-{K}-ORACLE", "GP-{K}-MLE", "RS", "UNK"],
                    help="prompt tag templates; 'RS-baseline' runs random search without the model")
    ap.add_argument("--hp_suffix", action="store_true")
    ap.add_argument("--subset", choices=["test_core", "test"], default="test_core")
    ap.add_argument("--fids", nargs="*", default=None, help="restrict to these fids")
    ap.add_argument("--max_functions", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--symmetries", type=int, nargs="+", default=[0], help="symmetries g to run (A1 invariance)")
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--no_space_first", dest="space_first", action="store_false")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    specs = load_specs(args.corpus)
    fids = sorted(f for f, s in specs.items() if s["split"] == "test" and (args.subset == "test" or s["test_core"]))
    if args.fids:
        fids = [f for f in fids if f in set(args.fids)]
    if args.max_functions:
        # balanced over families
        by_kind = {}
        for f in fids:
            by_kind.setdefault(specs[f]["kind"], []).append(f)
        per = max(1, args.max_functions // len(by_kind))
        fids = sorted(f for v in by_kind.values() for f in v[:per])
    inits = initial_designs(args.corpus, set(fids), set(args.seeds))
    corpus_of = {}
    for d in args.corpus:
        for f in fids:
            if (Path(d) / "functions" / f"{f}.npy").exists():
                corpus_of[f] = d

    jobs = [(f, s, t, g) for f in fids for s in args.seeds for t in args.tags for g in args.symmetries]
    i, n = map(int, args.shard.split("/"))
    jobs = jobs[i::n]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            try:
                d = json.loads(line)
                done.add((d["fid"], d["seed"], d["tag"], d["symmetry"]))
            except json.JSONDecodeError:
                pass
    jobs = [j for j in jobs if j not in done]
    student = None
    if any(t != "RS-baseline" for t in args.tags) and jobs:
        student = Student(args.checkpoint_dir, space_first=args.space_first)
    print(f"{len(jobs)} runs ({len(done)} already done)")

    t0 = time.time()
    with open(out, "a") as fo:
        for k, (fid, seed, tmpl, g) in enumerate(jobs):
            spec = specs[fid]
            f = GPGridFunction.from_corpus(corpus_of[fid], spec, symmetry=g)
            X0 = np.clip(apply_sym(inits[(fid, seed)], g), 0, 1)
            run_seed = seed_of("student", fid, seed, tmpl, g) % 2**31
            if tmpl == "RS-baseline":
                X, y, p_nn = random_search(f, X0, args.n_trials, run_seed)
                prompt_tag = None
            else:
                prompt_tag = resolve_tag(tmpl, spec, g, args.hp_suffix)
                X, y, p_nn = student.run(f, X0, prompt_tag, args.n_trials, run_seed)
            lr = f.log_regret(y)
            fo.write(json.dumps({
                "fid": fid, "family": spec["kind"], "seed": seed, "tag": tmpl, "prompt_tag": prompt_tag,
                "symmetry": g, "n_init": len(X0), "boundary": spec["boundary"],
                "edge_dist": edge_distance(spec["argmin"]), "X": np.round(X, 6).tolist(),
                "y": np.round(y, 6).tolist(), "log_regret": np.round(lr, 4).tolist(), "n_invalid": 0,
                "p_nonnumber": round(p_nn, 5)}) + "\n")
            fo.flush()
            if (k + 1) % 10 == 0:
                el = time.time() - t0
                print(f"{k + 1}/{len(jobs)} runs, {el / (k + 1):.1f}s per run")


if __name__ == "__main__":
    main()
