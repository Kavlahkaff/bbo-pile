"""Closed-loop rollouts of a trained model on the test functions, from the teachers' initial designs.
    python -m evaluation.rollout --checkpoint_dir ckpt/final --tags GP NM RS BEST UNK --output r.jsonl
All runs of a batch advance together, one trial at a time. Points are sampled token by token over the 1000 number tokens
(--temperature, 0 = greedy). Finished (fid, seed, tag) are skipped on resume; --shard i/n splits the work."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from datagen.functions import FAMILIES, Function, seed_of
from datagen.generate import BASE, N_TEST, SEEDS
from datagen.optimizers import TRIALS, initial_design
from evaluation.prompt import make_history
from open_optformer.history import Y_NORMS


def load_model(checkpoint_dir, device):
    from litgpt.config import Config
    from litgpt.model import GPT
    from litgpt.tokenizer import Tokenizer
    d = Path(checkpoint_dir)
    model = GPT(Config.from_file(d / "model_config.yaml"))
    sd = torch.load(d / "lit_model.pth", weights_only=True, map_location="cpu")
    model.load_state_dict(sd.get("model", sd))
    return model.to(device).eval(), Tokenizer(d)


class Student:
    def __init__(self, checkpoint_dir, y_norm, temperature=1.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tok = load_model(checkpoint_dir, self.device)
        self.numbers = torch.tensor([self.tok.token_to_id(str(i)) for i in range(1000)], device=self.device)
        self.comma = self.tok.token_to_id(",")
        self.y_norm, self.temperature = y_norm, temperature

    @torch.no_grad()
    def _sample(self, ids, gens):
        """One number per sequence (sequences of different length are right-padded; attention is causal)."""
        n = [len(s) for s in ids]
        x = torch.zeros((len(ids), max(n)), dtype=torch.long)
        for j, s in enumerate(ids):
            x[j, :n[j]] = torch.tensor(s)
        assert max(n) <= self.model.max_seq_length, "prompt longer than the model's context"
        logits = self.model(x.to(self.device))[torch.arange(len(ids)), torch.tensor(n) - 1][:, self.numbers].float().cpu()
        if self.temperature == 0:
            return logits.argmax(-1).tolist()
        p = torch.softmax(logits / self.temperature, -1)
        return [int(torch.multinomial(p[j], 1, generator=gens[j])) for j in range(len(ids))]

    def run_batch(self, jobs):
        """jobs: (Function, seed, tag). Returns per job (k, y)."""
        K = [np.rint(initial_design(f.i, s) * 999).astype(int).tolist() for f, s, _ in jobs]
        Y = [f(np.array(k) / 999).tolist() for (f, _, _), k in zip(jobs, K)]
        gens = [torch.Generator().manual_seed(seed_of("student", f.fid, s, t) % 2**31) for f, s, t in jobs]
        for _ in range(TRIALS):
            live = [j for j in range(len(jobs)) if len(K[j]) < TRIALS]
            if not live:
                break
            ids = [self.tok.encode(make_history(jobs[j][2], self.y_norm, K[j], Y[j]).get_prompt(), bos=False).tolist()
                   for j in live]
            k0 = self._sample(ids, [gens[j] for j in live])
            ids = [s + [int(self.numbers[a]), self.comma] for s, a in zip(ids, k0)]
            k1 = self._sample(ids, [gens[j] for j in live])
            for j, a, b in zip(live, k0, k1):
                K[j].append([a, b])
                Y[j].append(float(jobs[j][0](np.array([[a, b]]) / 999)[0]))
        return list(zip(K, Y))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--tags", nargs="+", default=["GP", "NM", "RS", "UNK"])
    ap.add_argument("--y_norm", choices=Y_NORMS, default="causal_log", help="must match the export")
    ap.add_argument("--functions", type=int, default=200, help="test functions per family")
    ap.add_argument("--seeds", type=int, default=SEEDS["test"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    assert a.functions <= N_TEST
    s, n = map(int, a.shard.split("/"))
    fns = [(fam, BASE + j) for j in range(a.functions) for fam in FAMILIES][s::n]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            try:
                d = json.loads(line)
                done.add((d["fid"], d["seed"], d["tag"]))
            except json.JSONDecodeError:
                pass
        if not open(out, "rb").read().endswith(b"\n"):
            open(out, "a").write("\n")                             # a line cut by a time limit
    student, t0 = Student(a.checkpoint_dir, a.y_norm, a.temperature), time.time()
    jobs = [(fam, i, sd, tag) for fam, i in fns for sd in range(a.seeds) for tag in a.tags
            if (f"{fam}-{i}", sd, tag) not in done]
    print(f"{len(jobs)} runs ({len(done)} done)", flush=True)
    cache = {}
    with open(out, "a") as fh:
        for b in range(0, len(jobs), a.batch):
            chunk = jobs[b:b + a.batch]
            cache = {key: cache.get(key) or Function(*key) for key in dict.fromkeys((f, i) for f, i, _, _ in chunk)}
            batch = [(cache[f, i], sd, tag) for f, i, sd, tag in chunk]
            for (fn, sd, tag), (k, y) in zip(batch, student.run_batch(batch)):
                fh.write(json.dumps({"fid": fn.fid, "family": fn.family, "group": fn.group, "i": fn.i, "seed": sd, "tag": tag,
                                     "n_init": len(initial_design(fn.i, sd)), "k": k, "y": y,
                                     "lr": np.round(fn.log_regret(y), 5).tolist()}) + "\n")
            fh.flush()
            print(f"  {min(b + a.batch, len(jobs))}/{len(jobs)}, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
