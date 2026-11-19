"""Figures and tables for figures/09_contrast. Reads ../raw/*.jsonl, writes PNG/CSV into ../ .
    python plot_contrast.py          screens A-D (2026-11 first round)
    python plot_contrast.py --new    A at scale, E, F, quantization (2026-11-19)"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap as LSC

from common import RAW, CatGP, G, base
from screen_b import levels

OUT = RAW.parent
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
GPC, TREE, THIRD, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
BLUE3 = ["#9ec5f4", "#5598e7", "#104281"]
BLUES = LSC.from_list("b", ["#104281", "#256abf", "#5598e7", "#9ec5f4", "#e8f1fc"])
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlesize": 10.5,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def per_fn(out, fam=None):
    R = pd.read_json(RAW / f"{out}.jsonl", lines=True)
    if fam is not None:
        R["regime"] = fam(R)
    return R.pivot_table(index=["regime", "i"], columns="teacher", values="auc_floor")


def gaps(P, rows, pairs):
    """pairs: list of (label, challenger, reference). Returns tidy frame of paired differences with CIs."""
    out = []
    for r in rows:
        Q = P.loc[r]
        for lab, a, b in pairs:
            d = (Q[a] - Q[b]).to_numpy()
            lo, hi = G.boot_ci(d)
            out.append({"family": r, "series": lab, "mean": d.mean(), "lo": lo, "hi": hi, "n": len(d)})
    return pd.DataFrame(out)


def dot_chart(D, rows, labels, colors, title, xlabel, name, note=None, height=None):
    series = list(dict.fromkeys(D.series))
    fig, ax = plt.subplots(figsize=(7.2, height or 0.55 * len(rows) * max(1, len(series) / 1.6) + 1.2))
    off = np.linspace(0.22, -0.22, len(series)) if len(series) > 1 else [0]
    for s, o, c in zip(series, off, colors):
        S = D[D.series == s].set_index("family").loc[rows]
        y = np.arange(len(rows))[::-1] + o
        ax.hlines(y, S.lo, S.hi, color=c, lw=2, zorder=2)
        ax.plot(S["mean"], y, "o", ms=8, mfc=c, mec=SURFACE, mew=2, ls="", zorder=3, label=s)
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    ax.set_yticks(np.arange(len(rows))[::-1], labels)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=24 if len(series) > 1 else 8)
    if len(series) > 1:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(series), handletextpad=0.3, columnspacing=1.2,
                  fontsize=9.5)
    if note:
        ax.text(1.0, -0.13 - 0.02 * (6 - len(rows)), note, transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=8.5)
    save(fig, name)


# ------------------------------------------------------------------------------------------- B: construction
def fig_construction():
    i, k = 6, 16
    spec, g = base(i)
    ro, rn = levels(i, k, "ord"), levels(i, k, "nom")
    lo, hi = g[ro].min(), g[ro].max()
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.5), gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.28})
    axs[0].imshow(g.T, origin="lower", extent=[0, 1, 0, 1], cmap=BLUES, aspect="auto", interpolation="nearest")
    for r in ro:
        axs[0].axvline(r / 511, color=TREE, lw=1.1)
    axs[0].set_title("1  GP sample, 16 slices")
    axs[0].set_xlabel("slice position (becomes the level)"); axs[0].set_ylabel("x0")
    for ax, rows, t in ((axs[1], ro, "2  ordinal: levels in slice order"), (axs[2], rn, "3  nominal: same slices, shuffled")):
        ax.imshow(g[rows].T, origin="lower", aspect="auto", extent=[-0.5, k - 0.5, 0, 1], cmap=BLUES, vmin=lo, vmax=hi,
                  interpolation="nearest")
        ax.set_title(t); ax.set_xlabel("level x1 (0..15)"); ax.set_xticks([0, 5, 10, 15])
    for ax in axs:
        ax.set_yticks([0, 1])
        for s in ax.spines.values():
            s.set_visible(False)
    save(fig, "03_categorical_construction")


# ------------------------------------------------------------------------------------------- B: identifiability
def ident_curve():
    T = pd.read_json(RAW / "screen_b_rs_traj.jsonl", lines=True)
    rows = []
    for k in (8, 16, 32):
        Tk = T[T.regime.str.startswith(f"k{k}_")]
        y = Tk.regime.str.endswith("ord").to_numpy()
        for n in (10, 15, 20, 30, 40):
            s = []
            for x, v in zip(Tk.X, Tk.y):
                X = np.array(x)[:n]; z = -np.array(v)[:n]; z = (z - z.mean()) / max(z.std(ddof=1), 1e-9)
                a, b = CatGP(X, z, k, "m32", True), CatGP(X, z, k, "m32", False)
                s.append(b._nll(b.theta) - a._nll(a.theta))
            acc = ((np.array(s) > 0) == y)
            m, lo, hi = acc.mean(), *G.boot_ci(acc.astype(float))
            rows.append({"k": k, "n": n, "acc": m, "lo": lo, "hi": hi})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "b_identify.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for k, c in zip((8, 16, 32), BLUE3):
        S = D[D.k == k]
        ax.fill_between(S.n, S.lo, S.hi, color=c, alpha=0.18, lw=0)
        ax.plot(S.n, S.acc, "-o", color=c, lw=2, ms=6, mec=SURFACE, mew=1.5, label=f"k = {k}")
        ax.text(S.n.iloc[-1] + 0.8, S.acc.iloc[-1], f"k = {k}", color=INK2, va="center", fontsize=9.5)
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (2, 2)))
    ax.text(10, 0.505, "chance", color=INK2, fontsize=9, va="bottom")
    ax.set_ylim(0.4, 1.0); ax.set_xlim(9, 45)
    ax.set_xlabel("observations seen (random-search prefix)"); ax.set_ylabel("share classified correctly")
    ax.set_title("Ordinal or nominal? Pick the GP with the higher marginal likelihood")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="lower right", ncol=3, fontsize=9)
    save(fig, "05_identify")
    return D


# ------------------------------------------------------------------------------------------- 2026-11-19 screens
def new_screens():
    """A at scale, E (no local structure), F (Rosenbrock vs local search), Q (quantization). Each part runs only when
    its raw file exists. Writes 1x_*.png and gaps_new.csv."""
    parts = []
    if (RAW / "screen_a_big.jsonl").exists():
        P = per_fn("screen_a_big")
        rows = ["gp", "n50_t5", "n50_t5_shuffled", "n200_t1_L16"]
        D = gaps(P, rows, [("ExtraTrees, tree spread", "et_ei", "m32"),
                           ("ExtraTrees, distance uncertainty κ = 0.5", "et_dist0.5", "m32"),
                           ("SMAC3-style forest", "smac3_trees", "m32")])
        n = P.loc[rows[0]].shape[0]
        dot_chart(D, rows, ["plain GP", "RF: 50 points, 5 trees", "RF: shuffled y", "RF: 1 tree, 16 leaves"],
                  [TREE, THIRD, GRAY], "A at scale  Tree teachers minus GP m32",
                  "difference in floored AUC (< 0: tree better)", "11_screen_a_big",
                  note=f"{n} functions per regime, 30 seeds; 95% paired bootstrap CI")
        parts.append(D.assign(screen="A_big"))
    if (RAW / "screen_e.jsonl").exists():
        P = per_fn("screen_e")
        rows = ["gp", "iid_m8", "iid_m32", "iid_m128", "iid_m512"]
        D = gaps(P, rows, [("GP m32", "m32", "rs"), ("(1+1)-ES", "es", "rs"), ("Nelder–Mead", "nm", "rs"),
                           ("ExtraTrees, distance unc.", "et_dist0.5", "rs")])
        dot_chart(D, rows, ["plain GP (control)", "iid cells 8 × 8", "iid cells 32 × 32", "iid cells 128 × 128",
                            "iid cells 512 × 512"], [GPC, THIRD, INK2, TREE],
                  "E  Teachers minus random search, functions without local structure",
                  "difference in floored AUC (> 0: random search better)", "12_screen_e",
                  note="160 functions per regime, 5 seeds; 95% paired bootstrap CI")
        parts.append(D.assign(screen="E"))
    if (RAW / "screen_f.jsonl").exists():
        P = per_fn("screen_f")
        rows = ["gp_T50", "rosen_T50", "gp_T200", "rosen_T200"]
        D = gaps(P, rows, [("(1+1)-ES", "es", "m32"), ("Nelder–Mead", "nm", "m32"), ("random search", "rs", "m32")])
        dot_chart(D, rows, ["plain GP, 50 trials", "Rosenbrock, 50 trials", "plain GP, 200 trials",
                            "Rosenbrock, 200 trials"], [THIRD, INK2, GRAY],
                  "F  Local search minus GP m32", "difference in floored AUC (< 0: better than GP)", "13_screen_f",
                  note="160 functions per regime, 5 seeds; Rosenbrock floor −6, GP floor −3")
        Dq = gaps(P, rows, [("(1+1)-ES: snapped − exact", "es_q", "es"), ("Nelder–Mead: snapped − exact", "nm_q", "nm")])
        dot_chart(Dq, rows, ["plain GP, 50 trials", "Rosenbrock, 50 trials", "plain GP, 200 trials",
                             "Rosenbrock, 200 trials"], [THIRD, INK2],
                  "Q  Cost of snapping x to the 1000-level token grid", "difference in floored AUC (> 0: snapping hurts)",
                  "14_quant_x", note="same random draws with and without snapping; 95% paired bootstrap CI")
        parts += [D.assign(screen="F"), Dq.assign(screen="Q")]
    for out in ("screen_e", "screen_f"):
        if (RAW.parent / f"quant_{out}.csv").exists():
            continue
        if (RAW / f"{out}_traj.jsonl").exists():
            import quant
            quant.main([out])
    if parts:
        pd.concat(parts).round(4).to_csv(OUT / "gaps_new.csv", index=False)


if __name__ == "__main__":
    import sys
    if "--new" in sys.argv:
        new_screens()
        sys.exit()
    # D: exploration level
    Pd = per_fn("screen_d")
    ls = pd.read_json(RAW / "screen_d.jsonl", lines=True).groupby("i").ls_geo.first()
    tert = pd.qcut(ls, 3, labels=["short", "mid", "long"])
    Pd.index = pd.MultiIndex.from_tuples([(f"ls {tert[i]}", i) for _, i in Pd.index])
    Dd = gaps(Pd, ["ls short", "ls mid", "ls long"], [("β = 0.25", "m32_b0.25", "m32_b4"), ("β = 1", "m32_b1", "m32_b4"),
                                                    ("β = 16", "m32_b16", "m32_b4")])
    dot_chart(Dd, ["ls short", "ls mid", "ls long"], ["short lengthscale", "mid lengthscale", "long lengthscale"],
              BLUE3, "D  GP-UCB m32 at other β, minus β = 4 (the corpus teacher)", "difference in floored AUC (< 0: better than β = 4)",
              "01_screen_d", note="160 GP functions, terciles of the geometric-mean lengthscale; 95% paired bootstrap CI")
    # A: trees
    Pa = per_fn("screen_a")
    rows_a = ["gp", "n50_t5", "n50_t5_shuffled", "n200_t1_L16"]
    Da = gaps(Pa, rows_a, [("ExtraTrees, tree spread (08 baseline)", "et_ei", "m32"),
                           ("ExtraTrees, distance uncertainty κ = 0.5", "et_dist0.5", "m32")])
    dot_chart(Da, rows_a, ["plain GP", "RF: 50 points, 5 trees", "RF: shuffled y", "RF: 1 tree, 16 leaves"],
              [TREE, THIRD], "A  Tree teachers minus GP m32", "difference in floored AUC (< 0: tree better)", "02_screen_a",
              note="40 functions per regime, 3 seeds; κ chosen on even-numbered functions")
    # B: categorical
    Pb = per_fn("screen_b")
    rows_b = ["k8_ord", "k8_nom", "k16_ord", "k16_nom", "k32_ord", "k32_nom"]
    Db = gaps(Pb, rows_b, [("GP, categorical kernel", "gpcat", "gpord"), ("TPE", "tpe", "gpord"),
                           ("ExtraTrees, distance unc.", "et_dist0.25", "gpord")])
    dot_chart(Db, rows_b, [f"k = {r.split('_')[0][1:]}, {'ordinal' if r.endswith('ord') else 'nominal'}" for r in rows_b],
              [GPC, THIRD, TREE], "B  Teachers minus the ordinal GP (levels treated as a number)",
              "difference in floored AUC (< 0: better than ordinal GP)", "04_screen_b",
              note="40 functions per family, 3 seeds; ordinal and nominal share the same slices")
    # C: noise
    Pc = per_fn("screen_c")
    rows_c = ["clean", "gauss0.05", "gauss0.2", "outlier0.1"]
    Dc = gaps(Pc, rows_c, [("GP m32, learned noise", "m32n", "m32"), ("TPE", "tpe", "m32"),
                           ("ExtraTrees, median, distance unc.", "tree_median", "m32")])
    dot_chart(Dc, rows_c, ["no noise", "Gaussian, σ = 0.05 sd", "Gaussian, σ = 0.2 sd", "10% outliers"],
              [GPC, THIRD, TREE], "C  Teachers minus the noise-free GP m32 (corpus teacher)",
              "difference in floored AUC (< 0: better than corpus teacher)", "06_screen_c",
              note="40 functions per regime, 3 seeds; regret on the noise-free function")
    pd.concat([Dd.assign(screen="D"), Da.assign(screen="A"), Db.assign(screen="B"), Dc.assign(screen="C")]).round(4).to_csv(
        OUT / "gaps.csv", index=False)
    means = {s: per_fn(s).groupby(level=0).mean().round(3) for s in ("screen_d", "screen_a", "screen_b", "screen_c")}
    with open(OUT / "means.json", "w") as fh:
        json.dump({s: m.to_dict(orient="index") for s, m in means.items()}, fh, indent=1)
    fig_construction()
    if not (OUT / "b_identify.csv").exists():
        print(ident_curve().round(3).to_string())
