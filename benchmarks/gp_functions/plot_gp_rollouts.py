"""
Trajectory pictures of the evaluation: per test function, the landscape with the queries of the teachers (oracle,
matched MLE, RS) and of every student (arm x prompt tag), plus the normalized regret over the trials.

    python benchmarks/gp_functions/plot_gp_rollouts.py --corpus corpus \
        --arms 'R=rollouts/R_shard_*.jsonl' 'O=rollouts/O_shard_*.jsonl' --tags 'GP-{K}-ORACLE' --output_path plots/

Functions (per family, --per_family of each category, judged on the first arm and first tag, floored AUC averaged
over seeds): "typical" (student-teacher gap closest to the family median), "worst" (largest gap), "best" (smallest
gap); plus --fids. Landscapes show seed --seed; regret curves show that seed (thin) and the mean over all seeds
(thick).

Output (--output_path): functions/<family>_<category>_<fid>.png, all_plots.pdf, selection.csv
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analyze_gp_eval import LF, TAG_TEACHER, teacher_label  # noqa: E402
from compare_gp_arms import ARM_COLORS, parse_arms  # noqa: E402

from open_optformer.gp_corpus import KINDS, edge_distance, iter_trajectories, load_specs  # noqa: E402

TEACHERS = [("oracle", "#222222", "--"), ("mle_matched", "#666666", "-."), ("rs", "#aaaaaa", ":")]


def load_runs(files, tags):
    """(fid, seed, tag) -> run dict, symmetry 0 only (last line wins)."""
    import json
    out = {}
    for p in files:
        for line in open(p):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d["symmetry"] == 0 and d["tag"] in tags:
                out[(d["fid"], d["seed"], d["tag"])] = d
    return out


def auc_floor(lr):
    return float(np.maximum(np.asarray(lr, float), LF).mean())


def select(runs, teachers, specs, tag, per_family, extra):
    """[(fid, category)] from the gap of the first arm (prompted with tag) to the teacher it imitates."""
    tname = TAG_TEACHER.get(tag, "oracle")
    gap = {}
    for (fid, seed, t), d in runs.items():
        if t != tag or (fid, seed, tname) not in teachers:
            continue
        gap.setdefault(fid, []).append(auc_floor(d["log_regret"]) - auc_floor(teachers[(fid, seed, tname)]["log_regret"]))
    G = pd.DataFrame([{"fid": f, "family": specs[f]["kind"], "gap": np.mean(v)} for f, v in gap.items()])
    chosen = []
    for k in KINDS:
        g = G[G.family == k].sort_values("gap")
        if g.empty:
            continue
        med = g.gap.median()
        cats = {"typical": g.iloc[(g.gap - med).abs().argsort()].fid.tolist(),
                "worst": g.fid.tolist()[::-1], "best": g.fid.tolist()}
        for cat, order in cats.items():
            n = 0
            for f in order:
                if n == per_family:
                    break
                if f not in {c[0] for c in chosen}:
                    chosen.append((f, cat))
                    n += 1
    chosen += [(f, "requested") for f in extra if f in gap and f not in {c[0] for c in chosen}]
    return chosen, G


def grid_of(fid, corpus):
    for d in corpus:
        p = Path(d) / "functions" / f"{fid}.npy"
        if p.exists():
            return np.load(p)
    raise FileNotFoundError(f"no grid for {fid}")


def landscape(ax, grid, spec, X, n_init, lr, title):
    ax.imshow(grid.T, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")  # rows of the grid = x0
    n = grid.shape[0]
    t = np.linspace(0, 1, n)
    ax.contour(t, t, grid.T, levels=10, colors="white", linewidths=0.35, alpha=0.5)
    X = np.asarray(X, float)
    ax.scatter(X[:n_init, 0], X[:n_init, 1], marker="s", s=26, facecolor="white", edgecolor="k", lw=0.7, zorder=3)
    sc = ax.scatter(X[n_init:, 0], X[n_init:, 1], c=np.arange(n_init, len(X)) + 1, cmap="plasma", vmin=1,
                    vmax=len(X), s=18, edgecolor="k", lw=0.3, zorder=4)
    a = spec["argmin"]
    ax.plot(a[0], a[1], marker="*", color="white", mec="k", ms=14, zorder=5)
    b = int(np.argmin(lr))
    ax.plot(X[b, 0], X[b, 1], "x", color="cyan", ms=10, mew=2.2, zorder=6)
    ax.set_title(f"{title}\nAUC floor {auc_floor(lr):.2f}, final {max(lr[-1], LF):.2f}", fontsize=8)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=6)
    return sc


def plot_function(fid, cat, spec, grid, panels, curves, seed, path_png, pdf):
    ncol = 4
    nrow_map = math.ceil(len(panels) / ncol)
    fig = plt.figure(figsize=(3.3 * ncol, 3.3 * nrow_map + 3.4))
    gs = fig.add_gridspec(nrow_map + 1, ncol, height_ratios=[1] * nrow_map + [1.05])
    sc = None
    for i, (name, d) in enumerate(panels):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        sc = landscape(ax, grid, spec, d["X"], d["n_init"], d["log_regret"], name)
    if sc is not None:
        cax = fig.add_axes([0.92, 0.45, 0.012, 0.4])
        fig.colorbar(sc, cax=cax, label="trial")
    ax = fig.add_subplot(gs[nrow_map, :])
    trials = np.arange(1, 51)
    for name, (color, ls, this, allseeds) in curves.items():
        if allseeds:
            ax.plot(trials, np.maximum(np.stack(allseeds), LF).mean(0), color=color, ls=ls, lw=2.2,
                    label=f"{name} (mean of {len(allseeds)} seeds)")
        if this is not None:
            ax.plot(trials, np.maximum(this, LF), color=color, ls=ls, lw=0.9, alpha=0.6)
    ax.axhline(LF, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("trial")
    ax.set_ylabel("log10 normalized regret (floored)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    fig.suptitle(f"{fid} ({spec['kind']}, {cat}; ls {spec['ls'][0]:.3f}/{spec['ls'][1]:.3f}, edge distance of the "
                 f"minimum {edge_distance(spec['argmin']):.3f}) - maps: seed {seed}, white squares = initial design, "
                 f"star = true minimum, x = best found; curves: thin = seed {seed}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.91, 0.97))
    fig.savefig(path_png, dpi=110)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="label=glob or label@TAG=glob (only that prompt; overrides --tags for this arm); the first "
                         "arm (and its first tag) selects the functions")
    ap.add_argument("--tags", nargs="+", default=["GP-{K}-ORACLE"])
    ap.add_argument("--per_family", type=int, default=1, help="functions per family and category")
    ap.add_argument("--fids", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0, help="seed whose queries are drawn")
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()
    out = Path(args.output_path)
    (out / "functions").mkdir(parents=True, exist_ok=True)

    arms = parse_arms(args.arms)
    arm_tags = {a: [a.partition("@")[2]] if "@" in a else list(args.tags) for a in arms}
    students = [(a, t) for a in arms for t in arm_tags[a]]
    if len(students) > len(ARM_COLORS):
        raise SystemExit(f"at most {len(ARM_COLORS)} students (arms x tags) per picture, got {len(students)}")
    runs = {a: load_runs(f, set(arm_tags[a])) for a, f in arms.items()}
    specs = load_specs(args.corpus)
    fids = {k[0] for r in runs.values() for k in r}
    teachers = {}
    for t in iter_trajectories(args.corpus, fields=("fid", "seed", "mid", "X", "log_regret", "n_init"), split="test"):
        if t["fid"] in fids:
            teachers[(t["fid"], t["seed"], teacher_label(t["mid"], specs[t["fid"]]["kind"]))] = t
    first = next(iter(runs))
    chosen, G = select(runs[first], teachers, specs, arm_tags[first][0], args.per_family, args.fids)
    G.to_csv(out / "student_gap_per_function.csv", index=False)
    rows = []
    with PdfPages(out / "all_plots.pdf") as pdf:
        for fid, cat in chosen:
            spec = specs[fid]
            panels, curves = [], {}
            def pick(store, key3):
                """The run of --seed, else of the smallest available seed (named in the panel title)."""
                d = store.get((fid, args.seed, key3))
                if d is not None:
                    return d, ""
                alt = sorted(s for (f, s, k) in store if f == fid and k == key3)
                return (store[(fid, alt[0], key3)], f" (seed {alt[0]})") if alt else (None, "")

            for name, color, ls in TEACHERS:
                d, note = pick(teachers, name)
                if d is not None:
                    panels.append((f"teacher {name}{note}", d))
                allseeds = [v["log_regret"] for (f, s, n), v in teachers.items() if f == fid and n == name]
                curves[f"teacher {name}"] = (color, ls, d and d["log_regret"], allseeds)
            for (a, tag), color in zip(students, ARM_COLORS):
                d, note = pick(runs[a], tag)
                label = f"{a.partition('@')[0]} [{tag.replace('{K}', spec['kind'].upper())}]"
                if d is not None:
                    panels.append((label + note, d))
                allseeds = [v["log_regret"] for (f, s, t), v in runs[a].items() if f == fid and t == tag]
                curves[label] = (color, "-", d and d["log_regret"], allseeds)
            png = out / "functions" / f"{spec['kind']}_{cat}_{fid}.png"
            plot_function(fid, cat, spec, grid_of(fid, args.corpus), panels, curves, args.seed, png, pdf)
            g = G[G.fid == fid]
            rows.append({"fid": fid, "family": spec["kind"], "category": cat,
                         "gap_auc_floor": float(g.gap.iloc[0]) if len(g) else np.nan, "png": png.name})
    pd.DataFrame(rows).to_csv(out / "selection.csv", index=False)
    print(f"{len(rows)} functions -> {out}")


if __name__ == "__main__":
    main()
