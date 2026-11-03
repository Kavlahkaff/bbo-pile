import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generate_training_data"))
import compile_gp_data as cgd  # noqa: E402
from open_optformer.gp_corpus import KINDS, apply_sym, swaps_axes  # noqa: E402
from open_optformer.history import History, quantize  # noqa: E402

MIDS = ["rs"] + [f"mle|ucb4|{k}" for k in KINDS]


def make_corpus(root: Path, n_per_split=(2, 2, 6)):
    """Tiny fake corpus: per kind n test, n val, n train functions; 2 seeds; 50 trials."""
    r = np.random.default_rng(0)
    specs, trajs = [], []
    for kind in KINDS:
        i = 0
        for split, n in zip(("test", "val", "train"), n_per_split):
            for _ in range(n):
                fid = f"{kind}-{i:04d}"
                i += 1
                a = r.uniform(size=2) if i % 3 else np.array([0.005, 0.5])  # every 3rd near the boundary
                specs.append({"fid": fid, "kind": kind, "split": split, "test_core": False,
                              "ls": [0.07, 0.3], "var": 1.5, "fmin": -2.0, "fmax": 2.0, "argmin": a.tolist(),
                              "boundary": bool(min(a.min(), 1 - a.max()) <= 0.02)})
                for seed in range(2):
                    for mid in MIDS + [f"true|ucb4|{kind}"]:
                        X = r.uniform(size=(50, 2))
                        y = r.uniform(-2, 2, size=50)
                        trajs.append({"key": f"{fid}|{seed}|{mid}", "fid": fid, "fkind": kind, "split": split,
                                      "seed": seed, "mid": mid, "X": X.round(6).tolist(), "y": y.round(6).tolist(),
                                      "log_regret": np.log10(np.minimum.accumulate((y + 2) / 4) + 1e-8).tolist()})
    root.mkdir(parents=True, exist_ok=True)
    json.dump(specs, open(root / "functions_000.json", "w"))
    with open(root / "traj_000.jsonl", "w") as f:
        for t in trajs:
            f.write(json.dumps(t) + "\n")
    return {s["fid"]: s for s in specs}, {t["key"]: t for t in trajs}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    specs, trajs = make_corpus(root)
    return root, specs, trajs


def export(corpus, tmp_path, *extra):
    root = corpus[0]
    out = tmp_path / "out"
    cgd.main(["--corpus", str(root), "--output_path", str(out), "--workers", "1", *extra])
    lines = {s: (out / f"{s}.txt").read_text().splitlines() for s in ("train", "valid")}
    return lines, json.load(open(out / "manifest.json"))


def tag_of(line):
    return re.search(r"algorithm:(.*?),history:", line).group(1)


def parse_history(line):
    body = line.split("history:")[1]
    pts = [p for p in body.split("|") if p]
    return np.array([[int(v) for v in p.split("*")[0].split(",")] for p in pts]), [int(p.split("*")[1]) for p in pts]


def test_default_prompt_unchanged():
    h = History(name="b", algorithm="RS", config_space=cgd.CONFIG_SPACE, remove_names=True)
    h.add_trial({"x0": 0.5, "x1": 0.25}, 1.0)
    assert h.get_prompt().startswith("algorithm:RS,search-space:{type:UNI")
    h2 = History(name="b", algorithm="RS", config_space=cgd.CONFIG_SPACE, remove_names=True, space_first=True)
    h2.add_trial({"x0": 0.5, "x1": 0.25}, 1.0)
    p = h2.get_prompt()
    assert p.startswith("search-space:{type:UNI") and ",algorithm:RS,history:500,250*0|" in p
    # identical apart from the position of the tag
    assert p.replace(",algorithm:RS", "") == h.get_prompt().replace("algorithm:RS,", "")


def test_test_split_never_exported_and_counts(corpus, tmp_path):
    root, specs, trajs = corpus
    lines, man = export(corpus, tmp_path)
    n_train = sum(1 for t in trajs.values() if specs[t["fid"]]["split"] == "train")
    n_val = sum(1 for t in trajs.values() if specs[t["fid"]]["split"] == "val")
    assert len(lines["train"]) == n_train and len(lines["valid"]) == n_val
    assert man["counts"]["train_trajectories"] == n_train
    # the exported lines reproduce exactly the train/val trajectories (x and y quantized as History does)
    train_X = {tuple(np.rint(np.asarray(t["X"][0]) * 999).astype(int)) for t in trajs.values()
               if specs[t["fid"]]["split"] == "train"}
    test_X = {tuple(np.rint(np.asarray(t["X"][0]) * 999).astype(int)) for t in trajs.values()
              if specs[t["fid"]]["split"] == "test"}
    first = {tuple(parse_history(l)[0][0]) for l in lines["train"]}
    assert first <= train_X and not (first & (test_X - train_X))


def test_tag_schemes():
    assert cgd.base_tag("true|ucb4|rbf", "kernel_mode") == "GP-RBF-ORACLE"
    assert cgd.base_tag("mle|ucb4|m12", "kernel_mode") == "GP-M12-MLE"
    assert cgd.base_tag("mle|ucb4|m12", "kernel") == cgd.base_tag("true|ucb4|m12", "kernel") == "GP-M12-UCB"
    assert cgd.base_tag("mle|ucb4|m52", "quality") == "GP-BASE"
    assert cgd.base_tag("rs", "quality") == "RS" and cgd.base_tag("rs", "generic") == "RS"
    assert cgd.base_tag("true|ucb4|m32", "generic") == "GP-UCB"
    assert cgd.base_tag("rs", "best") == "BEST"
    assert len(cgd.all_tags("kernel_mode")) == 9


def test_tag_dropout_rate(corpus, tmp_path):
    lines, _ = export(corpus, tmp_path, "--tag_dropout", "0.3")
    share = np.mean([tag_of(l) == "UNK" for l in lines["train"]])
    assert 0.2 < share < 0.4


def test_symmetries_and_hp_swap(corpus, tmp_path):
    root, specs, trajs = corpus
    lines, _ = export(corpus, tmp_path, "--symmetries", "8", "--hp_suffix")
    n_train = sum(1 for t in trajs.values() if specs[t["fid"]]["split"] == "train")
    assert len(lines["train"]) == 8 * n_train
    # ls of the fake spec: 0.07 -> small, 0.3 -> large; after an axis swap the order flips
    q_small = quantize(0.07, 0.05, 0.5, log_scale=True)
    q_large = quantize(0.3, 0.05, 0.5, log_scale=True)
    seen = set()
    by_key = {}
    for l in lines["train"]:
        m = re.search(r"hp:\{ls:(\d+),(\d+),os:(\d+)\}", l)
        seen.add((int(m.group(1)), int(m.group(2))))
    assert seen == {(q_small, q_large), (q_large, q_small)}
    # every transformed line is the quantization of g(X) for some g, with ls consistent with swaps_axes(g)
    t = next(t for t in trajs.values() if specs[t["fid"]]["split"] == "train")
    X = np.asarray(t["X"])
    cands = {g: np.rint(np.clip(apply_sym(X, g), 0, 1) * 999).astype(int) for g in range(8)}
    hits = []
    for l in lines["train"]:
        Q, _ = parse_history(l)
        for g, C in cands.items():
            if np.array_equal(Q, C):
                m = re.search(r"hp:\{ls:(\d+),(\d+)", l)
                assert (int(m.group(1)) == q_large) == swaps_axes(g)
                hits.append(g)
    assert sorted(hits) == list(range(8))


def test_permutation_rejected_with_symmetries(corpus, tmp_path):
    with pytest.raises(SystemExit):
        export(corpus, tmp_path, "--symmetries", "2", "--num_permutation", "1")
    with pytest.raises(SystemExit):
        export(corpus, tmp_path, "--hp_suffix", "--num_permutation", "1")


def test_boundary_filter_and_control(corpus, tmp_path):
    root, specs, _ = corpus
    _, man = export(corpus, tmp_path, "--drop_boundary")
    near = sum(1 for s in specs.values() if s["split"] == "train" and min(min(s["argmin"]), 1 - max(s["argmin"])) <= 0.02)
    n_train = sum(1 for s in specs.values() if s["split"] == "train")
    assert man["functions"]["train"] == n_train - near
    _, man_c = export(corpus, tmp_path, "--drop_boundary", "--boundary_control")
    assert man_c["functions"]["train"] == n_train - near


def test_frac_functions_stratified(corpus, tmp_path):
    _, man = export(corpus, tmp_path, "--frac_functions", "0.5")
    for k in KINDS:
        assert man["functions"]["by_family"][f"train/{k}"] == 3


def test_shorter_trajectories_train_only(corpus, tmp_path):
    root, specs, trajs = corpus
    lines, _ = export(corpus, tmp_path, "--sample_shorter_trajectories")
    n_train = sum(1 for t in trajs.values() if specs[t["fid"]]["split"] == "train")
    assert len(lines["train"]) == 5 * n_train
    assert sorted({len(parse_history(l)[1]) for l in lines["train"]}) == [1, 5, 10, 20, 50]
    assert {len(parse_history(l)[1]) for l in lines["valid"]} == {50}


def test_teacher_selection(corpus, tmp_path):
    lines, _ = export(corpus, tmp_path, "--teachers", "oracle")
    assert {tag_of(l) for l in lines["train"]} == {f"GP-{k.upper()}-ORACLE" for k in KINDS}
    lines, _ = export(corpus, tmp_path, "--teachers", "oracle,rs")
    assert "RS" in {tag_of(l) for l in lines["train"]}


def test_only_best(corpus, tmp_path):
    import pandas as pd
    root, specs, trajs = corpus
    rows = []
    for t in trajs.values():
        s = specs[t["fid"]]
        # make mle|ucb4|m32 best everywhere in nopriv, oracle best overall; noise per seed for the per-function rule
        base = {"rs": -0.5, "mle|ucb4|m32": -2.0}.get(t["mid"], -1.5 if t["mid"].startswith("mle") else -3.0)
        rows.append({"fid": t["fid"], "seed": t["seed"], "mid": t["mid"],
                     "teacher": "oracle" if t["mid"].startswith("true") else t["mid"],
                     "mode": t["mid"].split("|")[0] if t["mid"] != "rs" else "rs",
                     "family": s["kind"], "split": s["split"], "auc": base, "auc_floor": max(base, -3)})
    ranks = tmp_path / "ranks.parquet"
    pd.DataFrame(rows).to_parquet(ranks)
    lines, man = export(corpus, tmp_path, "--only_best", "nopriv", "--best_by", "family", "--ranks", str(ranks))
    assert {tag_of(l) for l in lines["train"]} == {"GP-M32-MLE"}
    assert set(man["only_best"]["family_best"].values()) == {"mle|ucb4|m32"}
    lines, _ = export(corpus, tmp_path, "--only_best", "all", "--best_by", "auc", "--ranks", str(ranks),
                      "--tag_scheme", "best")
    assert {tag_of(l) for l in lines["train"]} == {"BEST"}
    # all seeds of the kept teacher
    n_train_fn = sum(1 for s in specs.values() if s["split"] == "train")
    assert len(lines["train"]) == 2 * n_train_fn
