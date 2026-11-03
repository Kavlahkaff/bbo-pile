"""
Step 0 of the GP pipeline: rank the teachers per function, before any export or training.

Reads traj_*.jsonl (only key/fid/seed/mid/y/log_regret, last line wins) and functions_*.json of one or more corpus
directories and writes, to --output_path:

    teacher_ranks.parquet   one row per (fid, seed, mid): auc, auc_floor, auc_qfloor, final, final_floor, ...
    teacher_ranks_report.txt
    *.csv                   the tables of the report

Metrics (log10 normalised simple regret r_t, running best):
    auc          mean_t log10 r_t                                   (as auc_log in runs_*.jsonl)
    auc_floor    mean_t log10 max(r_t, 1e-3)                        (what 1000 quantization levels can resolve at best)
    auc_qfloor   mean_t log10 max(r_t, res)   res = (max y - min y) / 999 / (fmax - fmin) of this trajectory,
                 i.e. the per-trajectory resolution of bbo-pile's per-trajectory min/max y quantization

Report:
    1. teacher means per family
    2. how often each teacher is the per-function best (pool all, pool MLE+RS)
    3. selection headroom: best fixed vs per-function best, chosen on held-out seeds (no winner's curse)
    4. matched-kernel loss rate vs distance of the minimum to the edge (motivates A2, picks the margin)
    5. quantization floor: oracle - best MLE gap under auc / auc_floor / auc_qfloor
    6. per-function best teacher distribution, with and without the boundary filter (G3 x A2)

Usage:
    python rank_gp_teachers.py --corpus /path/corpus_student_40k [/path/corpus_student_40k_2] --output_path out/
"""
import argparse
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from open_optformer.gp_corpus import (FLOOR, KINDS, edge_distance, load_specs, parse_mid, read_traj_file,  # noqa: E402
                                      traj_files)

MARGINS = (0.01, 0.02, 0.05, 0.10)
METRICS = ("auc", "auc_floor", "auc_qfloor")
_SPECS = {}


def _rows_of_file(path):
    rows = []
    lf = math.log10(FLOOR)
    for t in read_traj_file(path, fields=("key", "fid", "seed", "mid", "y", "log_regret")).values():
        s = _SPECS[t["fid"]]
        lr = np.asarray(t["log_regret"], float)
        y = np.asarray(t["y"], float)
        res = (y.max() - y.min()) / 999 / (s["fmax"] - s["fmin"])
        lq = math.log10(max(res, 1e-12))
        m = parse_mid(t["mid"])
        rows.append({
            "fid": t["fid"], "seed": int(t["seed"]), "mid": t["mid"], "teacher": teacher_id(t["mid"]),
            "mode": m["mode"], "kernel": m["kernel"],
            "family": s["kind"], "split": s["split"], "test_core": bool(s.get("test_core", False)),
            "boundary": bool(s["boundary"]), "edge_dist": edge_distance(s["argmin"]),
            "matched": m["kernel"] == s["kind"],
            "auc": float(lr.mean()), "auc_floor": float(np.maximum(lr, lf).mean()),
            "auc_qfloor": float(np.maximum(lr, lq).mean()),
            "final": float(lr[-1]), "final_floor": float(max(lr[-1], lf)), "res": float(res),
        })
    return rows


def _init(specs):
    global _SPECS
    _SPECS = specs


def build_table(corpus_dirs, workers):
    specs = load_specs(corpus_dirs)
    files = traj_files(corpus_dirs)
    if workers > 1 and len(files) > 1:
        with mp.Pool(workers, initializer=_init, initargs=(specs,)) as pool:
            parts = pool.map(_rows_of_file, files)
    else:
        _init(specs)
        parts = [_rows_of_file(f) for f in files]
    df = pd.DataFrame([r for p in parts for r in p])
    dup = df.duplicated(["fid", "seed", "mid"])
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicate (fid, seed, mid) rows across files")
    return df


# ---------------------------------------------------------------------------- analyses
def teacher_id(mid):
    """Column used in per-function comparisons: the oracle exists only for the matched kernel, so every
    true|ucb4|<kind> maps to one 'oracle' column; MLE teachers and RS keep their mid."""
    return "oracle" if parse_mid(mid)["mode"] == "true" else mid


def pool_mask(df, pool):
    return df["mode"].isin(["mle", "rs"]) if pool == "nopriv" else np.ones(len(df), bool)


def per_function(df, metric, seeds=None):
    """(fid x mid) matrix of the metric, mean over the given seeds."""
    d = df if seeds is None else df[df.seed.isin(seeds)]
    return d.pivot_table(index="fid", columns="teacher", values=metric, aggfunc="mean")


def win_frequency(df, metric, pool):
    """Share of functions on which each teacher is best (mean over seeds); ties share the credit."""
    d = df[pool_mask(df, pool)]
    out = []
    for fam, g in d.groupby("family"):
        M = per_function(g, metric).dropna()
        best = M.values == M.values.min(1, keepdims=True)
        share = (best / best.sum(1, keepdims=True)).mean(0)
        out.append(pd.Series(share, index=M.columns, name=fam))
    W = pd.DataFrame(out).T.fillna(0.0)
    W["all"] = W.mean(axis=1)
    return W


def seed_splits(split):
    """(choose seeds, evaluate seeds) pairs: test chooses on 0-1, evaluates on 2-4; train/val use 0<->1."""
    return [([0, 1], [2, 3, 4])] if split == "test" else [([0], [1]), ([1], [0])]


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    b = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def headroom(df, metric, pool):
    """Selection headroom, per split and family. For every (choose, evaluate) seed pair:
        fixed      best fixed teacher over all functions of the split, chosen on `choose`
        family     best fixed teacher per (true) family, chosen on `choose`
        per_fn     per-function best teacher, chosen on `choose`        (upper bound for strategy selection)
        mixture    uniform random teacher (mean over teachers)
        in_sample  per-function best chosen and scored on `evaluate`    (optimistic, winner's curse)
    all scored on `evaluate`. gap = fixed - per_fn (positive = selection can help), with a bootstrap CI over
    functions."""
    d = df[pool_mask(df, pool)]
    fam_of = d.groupby("fid").family.first()
    rows = []
    for split, ds in d.groupby("split"):
        per_fid = []
        for choose, evaluate in seed_splits(split):
            C, E = per_function(ds, metric, choose), per_function(ds, metric, evaluate)
            fids = C.index.intersection(E.index)
            C, E = C.loc[fids].dropna(), E.loc[fids].dropna()
            fids = C.index.intersection(E.index)
            C, E = C.loc[fids], E.loc[fids]
            fixed = C.mean().idxmin()
            fam_best = {f: C[fam_of[C.index] == f].mean().idxmin() for f in fam_of[C.index].unique()}
            sel = C.idxmin(axis=1)
            r = pd.DataFrame({
                "family": fam_of[fids].values,
                "fixed": E[fixed].values,
                "family_fixed": [E.at[i, fam_best[fam_of[i]]] for i in fids],
                "per_fn": [E.at[i, sel[i]] for i in fids],
                "mixture": E.mean(axis=1).values,
                "in_sample": E.min(axis=1).values,
            }, index=fids)
            r["fixed_mid"] = fixed
            per_fid.append(r)
        R = pd.concat(per_fid).groupby(level=0).agg(
            {"family": "first", "fixed": "mean", "family_fixed": "mean", "per_fn": "mean", "mixture": "mean",
             "in_sample": "mean", "fixed_mid": "first"})
        for fam, g in list(R.groupby("family")) + [("all", R)]:
            gap = g.fixed - g.per_fn
            lo, hi = boot_ci(gap.values)
            rows.append({"split": split, "family": fam, "n_fn": len(g), "fixed_mid": g.fixed_mid.mode().iat[0],
                         **{k: g[k].mean() for k in ("fixed", "family_fixed", "per_fn", "mixture", "in_sample")},
                         "gap_fixed_minus_perfn": gap.mean(), "gap_ci_lo": lo, "gap_ci_hi": hi,
                         "gap_family_minus_perfn": (g.family_fixed - g.per_fn).mean()})
    return pd.DataFrame(rows)


def loss_rates(df, metric):
    """Loss rate of the matched kernel vs edge distance:
        oracle_loses   the oracle is not the best teacher of the full pool        (mean over seeds)
        mle_m_loses    the matched-kernel MLE teacher is not the best of the 4 MLE teachers"""
    M = per_function(df, metric)
    info = df.groupby("fid").agg(family=("family", "first"), edge=("edge_dist", "first"))
    info = info.loc[M.index]
    ora = M["oracle"]
    mle_cols = [c for c in M.columns if c.startswith("mle|")]
    mle_m = pd.Series({f: M.at[f, f"mle|ucb4|{info.at[f, 'family']}"] for f in M.index})
    oracle_loses = ora > M.min(axis=1) + 1e-12
    mle_loses = mle_m > M[mle_cols].min(axis=1) + 1e-12
    rows = []
    for m in MARGINS:
        near = info.edge <= m
        for fam in KINDS + ["all"]:
            sel = (info.family == fam) if fam != "all" else np.ones(len(info), bool)
            for where, w in (("near", near), ("interior", ~near)):
                s = sel & w
                rows.append({"margin": m, "family": fam, "where": where, "n_fn": int(s.sum()),
                             "share_fn": float(s.sum() / max(sel.sum(), 1)),
                             "oracle_loses": float(oracle_loses[s].mean()) if s.any() else np.nan,
                             "mle_matched_loses": float(mle_loses[s].mean()) if s.any() else np.nan})
    return pd.DataFrame(rows)


def quantization_floor(df):
    """Per family: oracle AUC, best MLE AUC (best fixed MLE teacher of the family) and their gap, for each metric."""
    rows = []
    for fam, g in df.groupby("family"):
        r = {"family": fam}
        for metric in METRICS:
            means = g.groupby("teacher")[metric].mean()
            ora = means["oracle"]
            best_mle = means[[c for c in means.index if c.startswith("mle|")]].min()
            r[f"oracle_{metric}"], r[f"best_mle_{metric}"], r[f"gap_{metric}"] = ora, best_mle, best_mle - ora
        r["gap_kept_floor"] = r["gap_auc_floor"] / r["gap_auc"] if r["gap_auc"] else np.nan
        r["gap_kept_qfloor"] = r["gap_auc_qfloor"] / r["gap_auc"] if r["gap_auc"] else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def best_distribution(df, metric, pool, margin=None):
    """Distribution of the per-function best teacher (mean over seeds) as used by --only_best, train+val only,
    optionally after dropping functions with the minimum within `margin` of the edge."""
    d = df[pool_mask(df, pool) & df.split.isin(["train", "val"])]
    if margin is not None:
        d = d[d.edge_dist > margin]
    M = per_function(d, metric).dropna()
    fam = d.groupby("fid").family.first().loc[M.index]
    best = M.idxmin(axis=1)
    T = pd.crosstab(best.rename("teacher"), fam, normalize="columns")
    T["all"] = best.value_counts(normalize=True)
    return T.fillna(0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True, help="one or more corpus directories")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()

    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    df = build_table(args.corpus, args.workers)
    df.to_parquet(out / "teacher_ranks.parquet", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 50)
    fmt = lambda x: x.round(3).to_string()
    say = [f"corpus: {args.corpus}",
           f"{df.fid.nunique()} functions, {len(df)} runs, {df.teacher.nunique()} teachers (oracle = matched kernel, true HPs); "
           f"splits: {df.groupby('split').fid.nunique().to_dict()}",
           "metric: log10 normalised simple regret, AUC = mean over the trajectory; floor = 1e-3; "
           "qfloor = per-trajectory y-quantization resolution"]

    T = df.pivot_table(index="teacher", columns="family", values=["auc", "auc_floor", "auc_qfloor", "final"])
    T.to_csv(out / "teacher_means.csv")
    say += ["", "== 1. TEACHER MEANS per family (all splits)", fmt(T)]

    say += ["", "== 2. WIN FREQUENCY (share of functions on which a teacher is best, mean over seeds)"]
    for pool in ("all", "nopriv"):
        for metric in ("auc", "auc_floor"):
            W = win_frequency(df, metric, pool)
            W.to_csv(out / f"win_freq_{pool}_{metric}.csv")
            say += [f"-- pool={pool}, metric={metric}", fmt(W)]

    say += ["", "== 3. SELECTION HEADROOM (chosen on held-out seeds; gap > 0 = per-function selection helps)"]
    for pool in ("all", "nopriv"):
        for metric in ("auc", "auc_floor"):
            H = headroom(df, metric, pool)
            H.to_csv(out / f"headroom_{pool}_{metric}.csv", index=False)
            say += [f"-- pool={pool}, metric={metric}", fmt(H.set_index(["split", "family"]))]

    say += ["", "== 4. MATCHED-KERNEL LOSS RATE vs edge distance of the minimum (metric auc)"]
    L = loss_rates(df, "auc")
    L.to_csv(out / "loss_rate_vs_margin.csv", index=False)
    say += [fmt(L[L.family == "all"].set_index(["margin", "where"])),
            "-- per family", fmt(L[L.family != "all"].set_index(["margin", "family", "where"]))]

    say += ["", "== 5. QUANTIZATION FLOOR: oracle vs best fixed MLE (gap = best MLE - oracle; kept = share of the raw "
                "gap that survives)"]
    Q = quantization_floor(df)
    Q.to_csv(out / "quantization_floor.csv", index=False)
    say += [fmt(Q.set_index("family"))]

    say += ["", "== 6. PER-FUNCTION BEST TEACHER (train+val, as --only_best would pick, metric auc)"]
    for pool in ("all", "nopriv"):
        for margin in (None, 0.02):
            B = best_distribution(df, "auc", pool, margin)
            B.to_csv(out / f"best_dist_{pool}_margin_{margin}.csv")
            say += [f"-- pool={pool}, boundary margin={margin}", fmt(B)]

    text = "\n".join(say)
    (out / "teacher_ranks_report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
