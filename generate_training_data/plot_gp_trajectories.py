"""
Visual investigation of the GP-BO teacher corpus (test split: the only split with stored function grids).

Loads every test trajectory (2,000 functions x 5 seeds x 6 teachers on the full corpus) and writes to --output_path:

    agg_regret_curves.png       median regret curve per teacher and family (IQR band), with the 1e-3 floor
    agg_final_regret.png        distribution of final regret per teacher and family vs the quantization floors
    agg_seed_consistency.png    is the per-function winner reproducible across seeds? (step-0 noise result)
    agg_oracle_gap_vs_edge.png  oracle advantage vs distance of the minimum to the edge (A2)
    agg_lengthscale_fit.png     fitted / true lengthscale of the MLE teachers over the run (over-smoothing)
    functions/<family>_<category>_<fid>.png
                                one page per selected function: landscape, regret curves of all teachers and seeds,
                                AUC per seed, the queries of every teacher on the landscape, distance of the queries to
                                the true minimum, fitted lengthscales
    all_plots.pdf               everything above in one file
    selection.csv               the selected functions and why

Functions are selected per family by category (typical, oracle far ahead, oracle behind, boundary, hardest, easiest,
noisiest across seeds) plus --random_per_family random ones and any --fids.

Usage:
    python plot_gp_trajectories.py --corpus /path/corpus_student_40k --output_path plots/ --workers 32
"""
import argparse
import json
import math
import multiprocessing as mp
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from open_optformer.gp_corpus import FLOOR, KINDS, edge_distance, load_specs, parse_mid, traj_files  # noqa: E402

TEACHERS = ["oracle", "mle|ucb4|rbf", "mle|ucb4|m52", "mle|ucb4|m32", "mle|ucb4|m12", "rs"]  # smooth -> rough
MLE = [t for t in TEACHERS if t.startswith("mle")]
COLORS = {"oracle": "black", "mle|ucb4|rbf": "tab:blue", "mle|ucb4|m52": "tab:green",
          "mle|ucb4|m32": "tab:orange", "mle|ucb4|m12": "tab:red", "rs": "tab:gray"}
LABELS = {"oracle": "oracle", "mle|ucb4|rbf": "MLE rbf", "mle|ucb4|m52": "MLE m52", "mle|ucb4|m32": "MLE m32",
          "mle|ucb4|m12": "MLE m12", "rs": "RS"}
LOG_FLOOR = math.log10(FLOOR)
EDGE_MARGIN = 0.02
SPLIT_RE = re.compile(r'"split":\s*"([a-z_]+)"')
CATEGORIES = {
    "typical": "oracle advantage over the best MLE teacher closest to the family median",
    "oracle_far_ahead": "largest oracle advantage over the best MLE teacher",
    "oracle_behind": "oracle worst relative to the best MLE teacher",
    "boundary": f"minimum within {EDGE_MARGIN:.0%} of an edge, typical oracle advantage",
    "hardest": "worst oracle AUC",
    "easiest": "best oracle AUC",
    "noisiest": "largest spread of the oracle AUC across seeds",
}


def teacher_of(mid):
    return "oracle" if parse_mid(mid)["mode"] == "true" else mid


def label(t, fkind=None):
    return LABELS[t] + (" *" if fkind is not None and t.endswith("|" + fkind) else "")


# ---------------------------------------------------------------------------- loading
def _load_file(path):
    """Test trajectories of one traj_*.jsonl (last line wins), with only the fields the plots need."""
    out = {}
    with open(path) as fh:
        for line in fh:
            m = SPLIT_RE.search(line)
            if m is None or m.group(1) != "test":
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            steps = d.get("steps") or []
            t = teacher_of(d["mid"])
            out[d["key"]] = {
                "fid": d["fid"], "seed": int(d["seed"]), "teacher": t, "n_init": int(d["n_init"]),
                "X": np.asarray(d["X"], np.float32), "y": np.asarray(d["y"], float),
                "lr": np.asarray(d["log_regret"], float),
                "ls": (np.asarray([s["hp"]["ls"] for s in steps], float) if t in MLE and steps
                       else np.empty((0, 2))),
                "n_fallback": sum(bool(s.get("fallback")) for s in steps),
            }
    return out


def load(corpus_dirs, workers):
    files = traj_files(corpus_dirs)
    if workers > 1 and len(files) > 1:
        with mp.Pool(workers) as pool:
            parts = pool.map(_load_file, files)
    else:
        parts = [_load_file(f) for f in files]
    T = {}  # fid -> teacher -> seed -> trajectory
    for p in parts:
        for tr in p.values():
            T.setdefault(tr["fid"], {}).setdefault(tr["teacher"], {})[tr["seed"]] = tr
    return T


def grid_of(fid, corpus_dirs):
    for d in corpus_dirs:
        p = Path(d) / "functions" / f"{fid}.npy"
        if p.exists():
            return np.load(p).astype(float)
    return None


def log_qres(tr, spec):
    """log10 of the per-trajectory y-quantization resolution in normalised-regret units (as auc_qfloor in step 0)."""
    res = (tr["y"].max() - tr["y"].min()) / 999 / (spec["fmax"] - spec["fmin"])
    return math.log10(max(res, 1e-12))


# ---------------------------------------------------------------------------- tables
def auc_table(T, specs):
    """One row per (fid, teacher, seed)."""
    rows = []
    for fid, by_t in T.items():
        s = specs[fid]
        for t, by_s in by_t.items():
            for seed, tr in by_s.items():
                rows.append({"fid": fid, "family": s["kind"], "teacher": t, "seed": seed,
                             "auc": tr["lr"].mean(), "auc_floor": np.maximum(tr["lr"], LOG_FLOOR).mean(),
                             "final": tr["lr"][-1], "qres": log_qres(tr, s),
                             "edge": edge_distance(s["argmin"])})
    return pd.DataFrame(rows)


def function_table(A):
    """One row per function: mean AUC per teacher, best MLE teacher, oracle advantage, seed spread of the oracle."""
    M = A.pivot_table(index="fid", columns="teacher", values="auc", aggfunc="mean")
    M = M.dropna(subset=[t for t in TEACHERS if t in M.columns])
    info = A.groupby("fid").agg(family=("family", "first"), edge=("edge", "first"))
    F = info.loc[M.index].copy()
    F["oracle_auc"] = M["oracle"]
    F["best_mle"] = M[MLE].idxmin(axis=1)
    F["best_mle_auc"] = M[MLE].min(axis=1)
    F["oracle_adv"] = F.best_mle_auc - F.oracle_auc  # > 0: oracle better
    F["oracle_sd"] = A[A.teacher == "oracle"].groupby("fid").auc.std().loc[M.index]
    return F


def select(F, per_family_random, extra_fids, seed=0):
    rng = np.random.default_rng(seed)
    picked = []
    for fam in KINDS:
        g = F[F.family == fam]
        if g.empty:
            continue
        used = set()

        def take(cat, idx):
            for fid in idx:
                if fid not in used:
                    used.add(fid)
                    picked.append({"fid": fid, "family": fam, "category": cat})
                    return

        med = g.oracle_adv.median()
        take("typical", (g.oracle_adv - med).abs().sort_values().index)
        take("oracle_far_ahead", g.oracle_adv.sort_values(ascending=False).index)
        take("oracle_behind", g.oracle_adv.sort_values().index)
        b = g[g.edge <= EDGE_MARGIN]
        if not b.empty:
            take("boundary", (b.oracle_adv - b.oracle_adv.median()).abs().sort_values().index)
        take("hardest", g.oracle_auc.sort_values(ascending=False).index)
        take("easiest", g.oracle_auc.sort_values().index)
        take("noisiest", g.oracle_sd.sort_values(ascending=False).index)
        rest = [f for f in g.index if f not in used]
        for fid in rng.permutation(rest)[:per_family_random]:
            take("random", [fid])
    for fid in extra_fids:
        if fid in F.index:
            picked.append({"fid": fid, "family": F.at[fid, "family"], "category": "manual"})
        else:
            print(f"warning: --fids {fid} is not a complete test function, skipped")
    S = pd.DataFrame(picked)
    return S.join(F, on="fid", rsuffix="_")


# ---------------------------------------------------------------------------- per-function page
def plot_function(fid, by_t, spec, grid, category, map_seed):
    fam = spec["kind"]
    teachers = [t for t in TEACHERS if t in by_t]
    seeds = sorted({s for t in teachers for s in by_t[t]})
    if map_seed not in seeds:
        map_seed = seeds[0]
    a0, a1 = spec["argmin"]
    ls0, ls1 = spec["ls"]

    fig = plt.figure(figsize=(20, 13.5))
    gs = fig.add_gridspec(3, 4, hspace=0.38, wspace=0.28)
    fig.suptitle(f"{fid}  [{category}: {CATEGORIES.get(category, category)}]\n"
                 f"true ls = ({ls0:.3f}, {ls1:.3f}), var = {spec['var']:.2f}, argmin = ({a0:.3f}, {a1:.3f}), "
                 f"edge distance = {edge_distance(spec['argmin']):.3f}, boundary flag = {spec['boundary']}    "
                 f"(* = matched kernel; maps and diagnostics: seed {map_seed})", fontsize=12)

    # landscape
    ax = fig.add_subplot(gs[0, 0])
    if grid is not None:
        im = ax.imshow(grid.T, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")  # rows of the grid = x0
        ax.contour(np.linspace(0, 1, grid.shape[0]), np.linspace(0, 1, grid.shape[1]), grid.T, 12,
                   colors="white", linewidths=0.4, alpha=0.6)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax.plot(a0, a1, "r*", ms=16, mec="white")
    ax.set(title="function (red star = global min)", xlabel="x0", ylabel="x1", xlim=(0, 1), ylim=(0, 1))

    # regret curves, all seeds
    ax = fig.add_subplot(gs[0, 1:3])
    trials = np.arange(1, 51)
    qres = []
    for t in teachers:
        L = np.array([by_t[t][s]["lr"] for s in seeds if s in by_t[t]])
        qres += [log_qres(by_t[t][s], spec) for s in seeds if s in by_t[t]]
        for row in L:
            ax.plot(trials[:len(row)], row, color=COLORS[t], alpha=0.25, lw=0.8)
        ax.plot(trials[:L.shape[1]], L.mean(0), color=COLORS[t], lw=2.2, label=label(t, fam),
                ls="--" if t == "rs" else "-")
    n_inits = [by_t[teachers[0]][s]["n_init"] for s in seeds if s in by_t[teachers[0]]]
    ax.axvspan(0.5, max(n_inits) + 0.5, color="0.9", zorder=0, label=f"initial design (n_init {min(n_inits)}-{max(n_inits)})")
    ax.axhline(LOG_FLOOR, color="k", ls=":", lw=1, label="1e-3 floor")
    ax.axhline(np.median(qres), color="tab:purple", ls=":", lw=1, label="median y-quantization resolution")
    ax.set(title="log10 normalised simple regret (running best): thin = seeds, thick = mean", xlabel="trial",
           ylabel="log10 regret", xlim=(1, 50))
    ax.legend(fontsize=8, ncol=2, loc="lower left")

    # AUC per seed
    ax = fig.add_subplot(gs[0, 3])
    for i, t in enumerate(teachers):
        v = np.array([by_t[t][s]["lr"].mean() for s in seeds if s in by_t[t]])
        ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, color=COLORS[t], s=25, zorder=3)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color=COLORS[t], lw=2.5)
    ax.set_xticks(range(len(teachers)), [label(t, fam) for t in teachers], rotation=40, ha="right", fontsize=9)
    ax.set(title="AUC per seed (bar = mean; lower = better)", ylabel="AUC")
    ax.grid(axis="y", alpha=0.3)

    # query maps
    slots = [gs[1, 0], gs[1, 1], gs[1, 2], gs[1, 3], gs[2, 0], gs[2, 1]]
    bg = grid[::4, ::4].T if grid is not None else None
    for t, slot in zip(teachers, slots):
        ax = fig.add_subplot(slot)
        tr = by_t[t].get(map_seed)
        if bg is not None:
            ax.imshow(bg, origin="lower", extent=(0, 1, 0, 1), cmap="Greys_r", alpha=0.55)
        if tr is not None:
            X, n0 = tr["X"], tr["n_init"]
            ax.scatter(X[:n0, 0], X[:n0, 1], marker="s", s=30, facecolor="white", edgecolor="k", lw=0.8,
                       label="initial design", zorder=3)
            sc = ax.scatter(X[n0:, 0], X[n0:, 1], c=np.arange(n0, len(X)) + 1, cmap="plasma", vmin=1, vmax=50,
                            s=22, edgecolor="k", lw=0.3, zorder=4)
            b = int(np.argmin(tr["y"]))
            ax.plot(X[b, 0], X[b, 1], "x", color="cyan", ms=12, mew=2.5, zorder=5, label=f"best found (trial {b + 1})")
            extra = f", {tr['n_fallback']} fallbacks" if tr["n_fallback"] else ""
            ax.set_title(f"{label(t, fam)}: AUC {tr['lr'].mean():.2f}, final {tr['lr'][-1]:.2f}{extra}", fontsize=10)
            if t == teachers[0]:
                fig.colorbar(sc, ax=ax, fraction=0.046, label="trial")
        ax.plot(a0, a1, "r*", ms=14, mec="white", zorder=6)
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
        if t == teachers[0]:
            ax.legend(fontsize=7, loc="upper right")

    # distance of each query to the true minimum
    ax = fig.add_subplot(gs[2, 2])
    for t in teachers:
        tr = by_t[t].get(map_seed)
        if tr is None:
            continue
        dist = np.hypot(tr["X"][:, 0] - a0, tr["X"][:, 1] - a1)
        ax.plot(np.arange(1, len(dist) + 1), dist, ".-", color=COLORS[t], lw=0.8, ms=4, label=label(t, fam))
    ax.axvline(by_t[teachers[0]][map_seed]["n_init"] + 0.5, color="0.5", ls=":")
    ax.set(yscale="log", title="distance of each query to the true min", xlabel="trial", ylabel="distance")
    ax.legend(fontsize=7, ncol=2)

    # fitted lengthscales
    ax = fig.add_subplot(gs[2, 3])
    for t in MLE:
        tr = by_t.get(t, {}).get(map_seed)
        if tr is None or not len(tr["ls"]):
            continue
        x = tr["n_init"] + 1 + np.arange(len(tr["ls"]))
        ax.plot(x, tr["ls"][:, 0] / ls0, color=COLORS[t], lw=1.5, label=label(t, fam))
        ax.plot(x, tr["ls"][:, 1] / ls1, color=COLORS[t], lw=1.5, ls="--")
    ax.axhline(1, color="k", lw=1)
    ax.set(yscale="log", title="MLE fitted / true lengthscale (solid ls0, dashed ls1)", xlabel="trial",
           ylabel="ratio", xlim=(1, 50))
    ax.legend(fontsize=7)
    return fig


# ---------------------------------------------------------------------------- aggregate figures
def per_trial(T, specs, fam, t, key="lr"):
    rows = []
    for fid, by_t in T.items():
        if specs[fid]["kind"] != fam or t not in by_t:
            continue
        L = np.array([tr[key] for tr in by_t[t].values()])
        rows.append(L.mean(0))
    return np.array(rows)


def fig_regret_curves(T, specs):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=False)
    trials = np.arange(1, 51)
    for ax, fam in zip(axes, KINDS):
        for t in TEACHERS:
            R = per_trial(T, specs, fam, t)
            if not len(R):
                continue
            lo, med, hi = np.percentile(R, [25, 50, 75], axis=0)
            ax.plot(trials, med, color=COLORS[t], lw=2, label=label(t, fam), ls="--" if t == "rs" else "-")
            ax.fill_between(trials, lo, hi, color=COLORS[t], alpha=0.1)
        ax.axhline(LOG_FLOOR, color="k", ls=":", lw=1)
        ax.set(title=f"{fam} ({len(R)} test functions)", xlabel="trial", ylabel="log10 regret", xlim=(1, 50))
        ax.legend(fontsize=8)
    fig.suptitle("Median over test functions of the seed-mean regret curve (band = IQR over functions; "
                 "dotted = 1e-3 floor; * = matched kernel)")
    return fig


def fig_final_regret(A):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)
    for ax, fam in zip(axes, KINDS):
        g = A[A.family == fam]
        data = [g[g.teacher == t].final.values for t in TEACHERS]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
        for patch, t in zip(bp["boxes"], TEACHERS):
            patch.set(facecolor=COLORS[t], alpha=0.35)
        ax.axhline(LOG_FLOOR, color="k", ls=":", lw=1.2, label="1e-3 floor")
        ax.axhline(g.qres.median(), color="tab:purple", ls=":", lw=1.2, label="median y-quant. resolution")
        below = {t: (g[g.teacher == t].final < LOG_FLOOR).mean() for t in TEACHERS}
        ax.set_xticks(range(1, len(TEACHERS) + 1),
                      [f"{label(t, fam)}\n{below[t]:.0%} < 1e-3" for t in TEACHERS], fontsize=8, rotation=30)
        ax.set(title=fam, ylabel="final log10 regret")
        ax.legend(fontsize=8)
    fig.suptitle("Final regret per run (all test seeds). Everything below the floors is invisible to a student that "
                 "sees 1000-level quantized y")
    return fig


def fig_seed_consistency(A):
    """Row 1: oracle advantage over the family's best fixed MLE teacher on seeds 0-1 vs 2-4.
       Row 2: winner on seeds 0-1 vs winner on seeds 2-4 (row-normalised)."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10.5))
    for j, fam in enumerate(KINDS):
        g = A[A.family == fam]
        if g.empty:
            continue
        fixed = g[g.teacher.isin(MLE)].groupby("teacher").auc.mean().idxmin()
        P = {name: g[g.seed.isin(s)].pivot_table(index="fid", columns="teacher", values="auc", aggfunc="mean")
             for name, s in (("a", [0, 1]), ("b", [2, 3, 4]))}
        fids = P["a"].dropna().index.intersection(P["b"].dropna().index)
        da = (P["a"].loc[fids, fixed] - P["a"].loc[fids, "oracle"]).values
        db = (P["b"].loc[fids, fixed] - P["b"].loc[fids, "oracle"]).values
        ax = axes[0, j]
        ax.scatter(da, db, s=6, alpha=0.4, color="k")
        lim = np.nanpercentile(np.abs(np.r_[da, db]), 99)
        ax.plot([-lim, lim], [-lim, lim], color="tab:red", lw=1)
        ax.axhline(0, color="0.6", lw=0.8), ax.axvline(0, color="0.6", lw=0.8)
        r = np.corrcoef(da, db)[0, 1] if len(da) > 2 else np.nan
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="seeds 0-1", ylabel="seeds 2-4",
               title=f"{fam}: AUC({LABELS[fixed]}) - AUC(oracle), r = {r:.2f}")
        teachers = [t for t in TEACHERS if t in P["a"].columns]
        wa = P["a"].loc[fids, teachers].idxmin(axis=1)
        wb = P["b"].loc[fids, teachers].idxmin(axis=1)
        C = pd.crosstab(wa, wb, normalize="index").reindex(index=teachers, columns=teachers).fillna(0)
        ax = axes[1, j]
        im = ax.imshow(C.values, cmap="Blues", vmin=0, vmax=1)
        for (i, k), v in np.ndenumerate(C.values):
            ax.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.5 else "k")
        names = [label(t, fam) for t in teachers]
        ax.set_xticks(range(len(teachers)), names, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(teachers)), names, fontsize=8)
        ax.set(title=f"{fam}: winner on seeds 0-1 (rows) vs 2-4 (cols); agree {np.mean(wa.values == wb.values):.0%}",
               xlabel="winner on seeds 2-4")
    fig.suptitle("Is the per-function result reproducible across seeds? Top: points on the red diagonal = stable "
                 "per-function difference, a round cloud = seed noise. Bottom: a strong diagonal = stable winner")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def fig_gap_vs_edge(F):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)
    bins = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    for ax, fam in zip(axes, KINDS):
        g = F[F.family == fam]
        ax.scatter(g.edge, g.oracle_adv, s=6, alpha=0.35, color="k")
        idx = np.digitize(g.edge, bins) - 1
        mid = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        mean = [g.oracle_adv[idx == i].mean() if (idx == i).any() else np.nan for i in range(len(bins) - 1)]
        ax.plot(mid, mean, "o-", color="tab:red", lw=2, label="binned mean")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(EDGE_MARGIN, color="tab:blue", ls="--", lw=1, label=f"{EDGE_MARGIN:.0%} margin")
        ax.set(title=f"{fam}: oracle advantage over best MLE (per function)", xlabel="edge distance of the min",
               ylabel="AUC(best MLE) - AUC(oracle)")
        ax.legend(fontsize=8)
    fig.suptitle("Oracle advantage vs distance of the minimum to the edge (> 0: oracle better; best MLE chosen per "
                 "function on the same seeds, so slightly optimistic for MLE)")
    return fig


def fig_lengthscale_fit(T, specs):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)
    trials = np.arange(1, 51)
    for ax, fam in zip(axes, KINDS):
        for t in MLE:
            rows = []
            for fid, by_t in T.items():
                s = specs[fid]
                if s["kind"] != fam or t not in by_t:
                    continue
                for tr in by_t[t].values():
                    if not len(tr["ls"]):
                        continue
                    r = np.full((50, 2), np.nan)
                    k = tr["n_init"] + np.arange(len(tr["ls"]))
                    k = k[k < 50]
                    r[k] = np.log10(tr["ls"][:len(k)] / np.asarray(s["ls"]))
                    rows += [r[:, 0], r[:, 1]]
            if not rows:
                continue
            R = np.array(rows)
            ok = np.sum(~np.isnan(R), 0) > 0.5 * len(R)
            lo, med, hi = np.nanpercentile(np.where(ok, R, np.nan), [25, 50, 75], axis=0)
            ax.plot(trials, 10 ** med, color=COLORS[t], lw=2, label=label(t, fam))
            ax.fill_between(trials, 10 ** lo, 10 ** hi, color=COLORS[t], alpha=0.1)
        ax.axhline(1, color="k", lw=1)
        ax.set(yscale="log", title=fam, xlabel="trial", ylabel="fitted / true lengthscale", xlim=(1, 50))
        ax.legend(fontsize=8)
    fig.suptitle("Online-MLE lengthscale fit over the run (median over runs and both dims, IQR band; > 1 = "
                 "over-smoothing)")
    return fig


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True, help="one or more corpus directories")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--random_per_family", type=int, default=2, help="extra random functions per family")
    ap.add_argument("--fids", nargs="*", default=[], help="extra test functions to plot, e.g. rbf-0012")
    ap.add_argument("--map_seed", type=int, default=0, help="seed shown in the query maps and diagnostics")
    ap.add_argument("--no_aggregate", action="store_true", help="only the per-function pages")
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    out = Path(args.output_path)
    (out / "functions").mkdir(parents=True, exist_ok=True)
    specs = load_specs(args.corpus)
    print("loading test trajectories ...", flush=True)
    T = load(args.corpus, args.workers)
    A = auc_table(T, specs)
    F = function_table(A)
    print(f"{len(T)} test functions, {len(A)} runs, {len(F)} with all teachers", flush=True)

    S = select(F, args.random_per_family, args.fids)
    S.to_csv(out / "selection.csv", index=False)

    with PdfPages(out / "all_plots.pdf") as pdf:
        def save(fig, path):
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print("wrote", path, flush=True)

        if not args.no_aggregate:
            save(fig_regret_curves(T, specs), out / "agg_regret_curves.png")
            save(fig_final_regret(A), out / "agg_final_regret.png")
            save(fig_seed_consistency(A), out / "agg_seed_consistency.png")
            save(fig_gap_vs_edge(F), out / "agg_oracle_gap_vs_edge.png")
            save(fig_lengthscale_fit(T, specs), out / "agg_lengthscale_fit.png")
        for r in S.itertuples():
            fig = plot_function(r.fid, T[r.fid], specs[r.fid], grid_of(r.fid, args.corpus), r.category,
                                args.map_seed)
            save(fig, out / "functions" / f"{r.family}_{r.category}_{r.fid}.png")


if __name__ == "__main__":
    main()
