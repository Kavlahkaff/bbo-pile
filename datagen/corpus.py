"""Reading the corpus."""
import json
from pathlib import Path


def read(corpus, split=None):
    files = sorted(Path(corpus).glob("traj_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no traj_*.jsonl in {corpus}")
    for f in files:
        for line in open(f):
            if split is None or f'"split": "{split}"' in line:
                yield json.loads(line)
