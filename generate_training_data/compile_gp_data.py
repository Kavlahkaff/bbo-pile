"""
Export GP-BO teacher trajectories (master-thesisv2/gp_traj.py corpus) to bbo-pile text lines.

One line per (trajectory, symmetry[, prefix]), built with open_optformer.history.History so quantization is identical
to the rest of bbo-pile:

    search-space:{type:UNI,min_value:0.0,max_value:1.0,linear_scale}{...},algorithm:<tag>,history:x0,x1*y|...

(--no_space_first gives the original order "algorithm:<tag>,search-space:...".)
train functions -> train.txt, val functions -> valid.txt. Test functions are never exported.
A manifest.json with all arguments, counts and commit hashes is written next to the output.
Lines are written in corpus order: shuffle afterwards (shuf), then preprocess_data.py.

Tag schemes (--tag_scheme):
    generic       GP-UCB / RS
    kernel        GP-RBF-UCB, GP-M12-UCB, ... / RS          (oracle and matched MLE share a tag)
    kernel_mode   GP-RBF-ORACLE / GP-RBF-MLE / ... / RS     (every teacher distinct; default)
    quality       GP-ORACLE / GP-BASE / RS
    best          BEST for everything exported
--hp_suffix appends ",hp:{ls:<q>,<q>,os:<q>}" (true task hyperparameters, log-quantized 0-999 over the generator
ranges; ls0/ls1 swapped when the symmetry swaps the axes) to any scheme. --tag_dropout p replaces the whole algorithm
string (incl. the hp suffix) by UNK.

Teacher selection:
    --teachers {oracle,mle,oracle+mle,nopriv,all} or a comma list of mids (true|ucb4|rbf, mle|ucb4|m32, rs, ...);
               "oracle" in a list means the matched-kernel oracle of each function; nopriv = the 4 MLE teachers + RS
    --only_best {all,nopriv} --best_by {auc,auc_floor,family,fixed}   needs --ranks teacher_ranks.parquet
        (rank_gp_teachers); pool all = every teacher, nopriv = MLE + RS (no oracle)
        per-function (auc/auc_floor): keep the teacher with the best mean over this function's seeds ("noisy best")
        family: keep the best fixed teacher per function family, chosen on train functions by auc_floor (the G3 B_np
                rule; with --only_best all this is the oracle everywhere, i.e. O)
        fixed:  keep one teacher for all functions, the best on the train functions by mean auc_floor
        all seeds of the kept teacher are exported.

Function selection (train/val; test is never touched):
    --boundary_margin m / --drop_boundary (m=0.02): drop functions whose minimum lies within m of an edge
    --boundary_control: instead drop the same number of random interior functions (A2 equal-count control)
    --frac_functions f --subset_seed s: keep a fraction of the train functions, stratified by family

Augmentation:
    --symmetries k [--symmetry_set d4|noswap|rot]: k of the square symmetries per trajectory (identity always included)
    --prefix_mode {none,fixed,random} (train only; valid.txt always has full trajectories only):
        fixed   also the first [1, 5, 10, 20] trials as extra lines (as compile_data.py;
                --sample_shorter_trajectories is the old spelling of this)
        random  also --n_prefixes prefixes with lengths drawn uniformly (without replacement) from
                [n_init + 1, n - 1], i.e. every prefix contains at least one BO step; seeded per trajectory
        none    full trajectories only (default)
        Prefix lines matter with --y_norm global (they are the only lines whose y scaling matches inference);
        with --y_norm causal every prefix of a full line is already exactly an inference prompt.
    --y_norm {global,causal}: y quantization, see History.y_norm (rollouts / scoring must use the same)
    --num_permutation n: hyperparameter-order shuffles (default 0). With two identical UNI dims this IS the axis swap,
        so it is rejected together with --symmetries > 1 or --hp_suffix.
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syne_tune.config_space import uniform  # noqa: E402

from open_optformer.gp_corpus import (KINDS, SYMMETRY_SETS, apply_sym, git_commit, is_oracle, load_specs,  # noqa: E402
                                      near_boundary, parse_mid, quantize_hp, read_traj_file, rng_of,
                                      sample_symmetries, traj_files)
from open_optformer.history import History  # noqa: E402

SHORTER = (1, 5, 10, 20)
SPLIT_FILE = {"train": "train", "val": "valid"}
CONFIG_SPACE = {"x0": uniform(0.0, 1.0), "x1": uniform(0.0, 1.0)}
UNK = "UNK"


# ---------------------------------------------------------------------------- tags
def base_tag(mid: str, scheme: str) -> str:
    m = parse_mid(mid)
    if scheme == "best":
        return "BEST"
    if m["mode"] == "rs":
        return "RS"
    k = m["kernel"].upper()
    if scheme == "generic":
        return "GP-UCB"
    if scheme == "kernel":
        return f"GP-{k}-UCB"
    if scheme == "kernel_mode":
        return f"GP-{k}-{'ORACLE' if m['mode'] == 'true' else 'MLE'}"
    if scheme == "quality":
        return "GP-ORACLE" if m["mode"] == "true" else "GP-BASE"
    raise ValueError(f"unknown tag scheme {scheme}")


def hp_suffix(spec: dict, g: int) -> str:
    l0, l1, o = quantize_hp(spec, g)
    return f",hp:{{ls:{l0},{l1},os:{o}}}"


def make_tag(mid, spec, g, args, key) -> str:
    tag = base_tag(mid, args.tag_scheme)
    if args.hp_suffix:
        tag += hp_suffix(spec, g)
    if args.tag_dropout > 0 and rng_of("dropout", args.seed, key, g).uniform() < args.tag_dropout:
        tag = UNK
    return tag


def all_tags(scheme: str, with_hp: bool = False) -> list[str]:
    """Every distinct base tag of a scheme (for the tokenizer symbol list and the manifest)."""
    mids = ["rs"] + [f"{mode}|ucb4|{k}" for mode in ("true", "mle") for k in KINDS]
    return sorted({base_tag(m, scheme) for m in mids})


# ---------------------------------------------------------------------------- selection
def select_functions(specs: dict, args) -> dict[str, dict]:
    """fid -> spec of the exported (train/val) functions after boundary filter / control and subsampling."""
    keep = {f: s for f, s in specs.items() if s["split"] in SPLIT_FILE}
    margin = 0.02 if args.drop_boundary else args.boundary_margin
    if margin is not None:
        for split in SPLIT_FILE:
            for kind in KINDS:
                grp = sorted(f for f, s in keep.items() if s["split"] == split and s["kind"] == kind)
                near = [f for f in grp if near_boundary(keep[f], margin)]
                if args.boundary_control:
                    interior = [f for f in grp if f not in set(near)]
                    r = rng_of("boundary_control", args.subset_seed, split, kind)
                    drop = r.choice(interior, size=min(len(near), len(interior)), replace=False).tolist()
                else:
                    drop = near
                for f in drop:
                    keep.pop(f)
    if args.frac_functions < 1.0:
        for kind in KINDS:
            grp = sorted(f for f, s in keep.items() if s["split"] == "train" and s["kind"] == kind)
            n = min(len(grp), max(1, int(round(args.frac_functions * len(grp)))))  # at least one per family
            r = rng_of("frac", args.subset_seed, kind)
            for f in set(grp) - set(r.choice(grp, size=n, replace=False).tolist()):
                keep.pop(f)
    return keep


def teacher_filter(teachers: str):
    """Predicate (mid, spec) -> bool for --teachers."""
    if teachers == "all":
        return lambda mid, s: True
    if teachers == "oracle":
        return lambda mid, s: is_oracle(mid)
    if teachers == "mle":
        return lambda mid, s: parse_mid(mid)["mode"] == "mle"
    if teachers == "oracle+mle":
        return lambda mid, s: parse_mid(mid)["mode"] in ("true", "mle")
    if teachers == "nopriv":
        return lambda mid, s: parse_mid(mid)["mode"] in ("mle", "rs")
    wanted = {t.strip() for t in teachers.split(",")}
    return lambda mid, s: mid in wanted or ("oracle" in wanted and is_oracle(mid))


FIXED_METRIC = "auc_floor"  # metric of the fixed / per-family choices


def load_ranks(ranks_path, pool: str):
    import pandas as pd
    df = pd.read_parquet(ranks_path)
    if pool == "nopriv":
        df = df[df["mode"].isin(["mle", "rs"])]
    return df


def fixed_best_mid(ranks_path, pool: str = "nopriv") -> str:
    """The single best teacher on the TRAIN functions (mean auc_floor over functions and seeds). Used by the export
    (--best_by fixed) and by gonogo.sh for the rollout tag of that teacher, so both use the same rule."""
    tr = load_ranks(ranks_path, pool)
    tr = tr[tr.split == "train"]
    best = tr.groupby("teacher")[FIXED_METRIC].mean().idxmin()
    if best == "oracle":
        raise ValueError("the best fixed teacher is the oracle (pool all), which has no single mid; use pool nopriv")
    return best


def best_teachers(ranks_path, pool: str, best_by: str, fids) -> tuple[dict[str, str], dict]:
    """fid -> the single mid to keep, plus info for the manifest."""
    df = load_ranks(ranks_path, pool)
    if best_by == "fixed":
        mid = fixed_best_mid(ranks_path, pool)
        means = df[df.split == "train"].groupby("teacher")[FIXED_METRIC].mean().sort_values()
        return {f: mid for f in fids}, {"fixed": mid, "train_means": means.round(4).to_dict()}
    if best_by == "family":
        # best fixed teacher per family, chosen on TRAIN functions only (never on val/test)
        tr = df[df.split == "train"]
        means = tr.groupby(["family", "teacher"])[FIXED_METRIC].mean().unstack()
        fam_best = means.idxmin(axis=1).to_dict()
        fam_of = df.groupby("fid").family.first()
        out = {}
        for f in fids:
            t = fam_best[fam_of[f]]
            out[f] = f"true|ucb4|{fam_of[f]}" if t == "oracle" else t
        return out, {"family_best": fam_best, "family_means": means.round(4).to_dict()}
    d = df[df.fid.isin(list(fids))]
    m = d.groupby(["fid", "mid"])[best_by].mean().reset_index()
    best = m.loc[m.groupby("fid")[best_by].idxmin()].set_index("fid")["mid"].to_dict()
    return best, {"best_distribution": Counter(best.values()).most_common()}


# ---------------------------------------------------------------------------- lines
def prefix_lengths(t: dict, args) -> list[int]:
    """Lengths of the extra prefix lines of one (train) trajectory, the same for all its symmetries."""
    n = len(t["y"])
    if args.prefix_mode == "fixed":
        return [m for m in SHORTER if m < n]
    if args.prefix_mode == "random":
        lo = int(t.get("n_init", 1)) + 1
        pool = np.arange(lo, n)
        if len(pool) == 0:
            return []
        r = rng_of("prefix", args.seed, t["key"])
        return sorted(int(m) for m in r.choice(pool, size=min(args.n_prefixes, len(pool)), replace=False))
    return []


def trajectory_lines(t: dict, spec: dict, args, train: bool) -> list[str]:
    X = np.asarray(t["X"], float)
    y = list(map(float, t["y"]))
    lines = []
    for g in sample_symmetries(args.symmetries, t["key"], args.seed, args.symmetry_set):
        Xg = np.clip(apply_sym(X, g), 0.0, 1.0)
        tag = make_tag(t["mid"], spec, g, args, t["key"])
        lengths = [len(y)] + (prefix_lengths(t, args) if train else [])
        for n in lengths:
            h = History(name=t["fid"], algorithm=tag, config_space=CONFIG_SPACE, remove_names=True,
                        space_first=args.space_first, y_norm=args.y_norm)
            for (x0, x1), v in zip(Xg[:n], y[:n]):
                h.add_trial({"x0": float(x0), "x1": float(x1)}, v)
            lines.append(h.get_prompt())
            if n == len(y):
                lines += [h.get_prompt(shuffle=True) for _ in range(args.num_permutation)]
    return lines


_G = {}


def _init(state):
    _G.update(state)


def _process_file(job):
    idx, path = job
    args, keep, allowed = _G["args"], _G["keep"], _G["allowed"]
    tmp = Path(_G["tmp"])
    counts = Counter()
    handles = {s: open(tmp / f"{SPLIT_FILE[s]}_{idx:05d}.txt", "w", encoding="utf-8") for s in SPLIT_FILE}
    try:
        for t in read_traj_file(path, fields=("key", "fid", "seed", "mid", "X", "y", "n_init")).values():
            spec = keep.get(t["fid"])
            if spec is None or not allowed(t["mid"], spec):
                continue
            split = spec["split"]
            assert split in SPLIT_FILE, f"refusing to export {t['fid']} of split {split}"
            lines = trajectory_lines(t, spec, args, train=split == "train")
            for line in lines:
                handles[split].write(line + "\n")
            counts[f"{split}_trajectories"] += 1
            counts[f"{split}_lines"] += len(lines)
            counts[f"{split}_mid={t['mid']}"] += 1
            counts[f"{split}_tag={'UNK' if lines[0].split('algorithm:')[1].startswith(UNK + ',') else 'tagged'}"] += 1
    finally:
        for h in handles.values():
            h.close()
    return counts


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", nargs="+", required=True, help="one or more corpus directories")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--tag_scheme", default="kernel_mode", choices=["generic", "kernel", "kernel_mode", "quality", "best"])
    ap.add_argument("--hp_suffix", action="store_true")
    ap.add_argument("--tag_dropout", type=float, default=0.0)
    ap.add_argument("--teachers", default="all")
    ap.add_argument("--only_best", choices=["all", "nopriv"], default=None)
    ap.add_argument("--best_by", choices=["auc", "auc_floor", "family", "fixed"], default="family")
    ap.add_argument("--ranks", default=None, help="teacher_ranks.parquet (needed for --only_best)")
    ap.add_argument("--boundary_margin", type=float, default=None)
    ap.add_argument("--drop_boundary", action="store_true", help="shorthand for --boundary_margin 0.02")
    ap.add_argument("--boundary_control", action="store_true",
                    help="drop as many random interior functions as the boundary filter would drop")
    ap.add_argument("--symmetries", type=int, default=1)
    ap.add_argument("--symmetry_set", choices=list(SYMMETRY_SETS), default="d4")
    ap.add_argument("--frac_functions", type=float, default=1.0)
    ap.add_argument("--subset_seed", type=int, default=0)
    ap.add_argument("--prefix_mode", choices=["none", "fixed", "random"], default=None,
                    help="extra prefix lines per train trajectory (default none)")
    ap.add_argument("--n_prefixes", type=int, default=2, help="prefixes per trajectory for --prefix_mode random")
    ap.add_argument("--sample_shorter_trajectories", action="store_true", help="= --prefix_mode fixed")
    ap.add_argument("--y_norm", choices=["global", "causal"], default="global")
    ap.add_argument("--num_permutation", type=int, default=0)
    ap.add_argument("--no_space_first", dest="space_first", action="store_false")
    ap.add_argument("--seed", type=int, default=0, help="seed for symmetry sampling and tag dropout")
    ap.add_argument("--generator_commit", default=None, help="override if the corpus is not inside a git checkout")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args(argv)
    if args.num_permutation > 0 and (args.symmetries > 1 or args.hp_suffix):
        ap.error("--num_permutation swaps x0/x1 (an axis symmetry) and does not swap the hp lengthscales; "
                 "use --symmetries instead")
    if args.only_best and not args.ranks:
        ap.error("--only_best needs --ranks teacher_ranks.parquet")
    if (args.boundary_control) and args.boundary_margin is None and not args.drop_boundary:
        ap.error("--boundary_control needs --boundary_margin or --drop_boundary")
    if args.sample_shorter_trajectories:
        if args.prefix_mode not in (None, "fixed"):
            ap.error("--sample_shorter_trajectories means --prefix_mode fixed")
        args.prefix_mode = "fixed"
    args.prefix_mode = args.prefix_mode or "none"
    if args.n_prefixes < 1:
        ap.error("--n_prefixes must be >= 1")
    if not 0 < args.frac_functions <= 1:
        ap.error("--frac_functions must be in (0, 1]")
    return args


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    specs = load_specs(args.corpus)
    keep = select_functions(specs, args)

    tfilter = teacher_filter(args.teachers)
    best_info = None
    if args.only_best:
        best, best_info = best_teachers(args.ranks, args.only_best, args.best_by, keep.keys())
        missing = set(keep) - set(best)
        if missing:
            raise ValueError(f"{len(missing)} functions have no ranking, e.g. {sorted(missing)[:3]}")
        allowed_mid = best
        allowed = lambda mid, s: allowed_mid[s["fid"]] == mid and tfilter(mid, s)
    else:
        allowed = tfilter

    tmp = out / "_parts"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    jobs = list(enumerate(traj_files(args.corpus)))
    state = {"args": args, "keep": keep, "allowed": allowed, "tmp": str(tmp)}
    if args.workers > 1 and len(jobs) > 1:
        # fork keeps the lambdas; the corpus is read once per file
        with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(state,)) as pool:
            parts = pool.map(_process_file, jobs)
    else:
        _init(state)
        parts = [_process_file(j) for j in jobs]
    counts = sum(parts, Counter())

    for split, name in SPLIT_FILE.items():
        with open(out / f"{name}.txt", "w", encoding="utf-8") as fo:
            for idx, _ in jobs:
                with open(tmp / f"{name}_{idx:05d}.txt", encoding="utf-8") as fi:
                    shutil.copyfileobj(fi, fo)
    shutil.rmtree(tmp)

    n_fn = Counter(s["split"] for s in keep.values())
    manifest = {
        "args": vars(args),
        "functions": {"train": n_fn["train"], "val": n_fn["val"],
                      "by_family": dict(Counter(f"{s['split']}/{s['kind']}" for s in keep.values()))},
        "counts": dict(sorted(counts.items())),
        "tags": all_tags(args.tag_scheme),
        "val_note": "valid.txt uses the same tags, dropout, symmetries and y_norm as train.txt, but no prefix lines",
        "only_best": best_info,
        "commits": {"bbo-pile": git_commit(Path(__file__).parent),
                    "generator": args.generator_commit or {str(c): git_commit(c) for c in args.corpus}},
        "example": open(out / "train.txt").readline().strip()[:400],
    }
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1, default=str)
    print(json.dumps({k: manifest[k] for k in ("functions", "counts", "tags", "only_best")}, indent=1, default=str))


if __name__ == "__main__":
    main()
