"""
Analysis of student rollouts (run_gp_benchmark.py) and teacher-forced scores (score_teacher_traces.py) of ONE model.
Several models / settings against each other: compare_gp_arms.py; trajectory pictures: plot_gp_rollouts.py.

Metrics: log10 normalised simple regret (fmin / fmax of the spec), AUC = mean over the 50 trials, raw and floored at
1e-3. Comparisons are paired per (function, seed), averaged per function, with bootstrap CIs over functions; every
table is given for all functions, per family, and split boundary / interior.
"closed" = (student - RS) / (teacher - RS) on the group means: the share of the RS -> teacher gap the student closes
(1 = as good as the teacher, 0 = random search).

    python benchmarks/gp_functions/analyze_gp_eval.py --corpus corpus --rollouts rollouts.jsonl \
        --scores scores.parquet --output_path eval/

Sections
    1. student (per prompt tag) vs each teacher it should imitate + the go/no-go comparison
       student prompted as oracle vs the matched-kernel MLE teacher (floored)
    1b. student vs student of the same model (oracle tag vs MLE tag vs UNK), paired
    1c. regret vs trial (floored), per family: regret_curves.csv / regret_curves.png
    2. tag-likelihood confusion matrix: rows = teacher of the trace, columns = prompt tag (relative to the function
       family), entry = mean per-step NLL of the teacher's next x over the BO steps; plus next-x distance views
       (mean, median) and hit rates (greedy next x within 0.02 / 0.05 of the teacher's)
    3. go/no-go verdict
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from open_optformer.gp_corpus import FLOOR, KINDS, edge_distance, iter_trajectories, load_specs  # noqa: E402

LF = math.log10(FLOOR)

# which teacher a prompt tag template should imitate (kernel_mode scheme)
TAG_TEACHER = {"GP-{K}-ORACLE": "oracle", "GP-{K}-MLE": "mle_matched", "RS": "rs", "RS-baseline": "rs"}
# fixed colours: a student and the teacher it imitates share the hue (student solid, teacher dashed)
TAG_COLOR = {"GP-{K}-ORACLE": "#d62728", "GP-{K}-MLE": "#1f77b4", "RS": "#7f7f7f", "UNK": "#9467bd",
             "RS-baseline": "#bcbd22"}
TEACHER_COLOR = {"oracle": "#d62728", "mle_matched": "#1f77b4", "rs": "#7f7f7f"}


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    r = np.random.default_rng(seed)
    b = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def teacher_label(mid, family):
    if mid == "rs":
        return "rs"
    mode, _, k = mid.split("|")
    if mode == "true":
        return "oracle"
    return "mle_matched" if k == family else f"mle|{k}"


def run_metrics(lr):
    lr = np.asarray(lr, float)
    return {"auc": float(lr.mean()), "auc_floor": float(np.maximum(lr, LF).mean()), "final": float(lr[-1]),
            "final_floor": float(max(lr[-1], LF))}


def repeat_rate(X, n_init):
    """Share of the model's proposals (after the initial design) that repeat an earlier point of the run on the
    1000-level grid: high values mean the policy is stuck (typical failure of greedy decoding)."""
    q = [tuple(np.rint(np.asarray(x) * 999).astype(int)) for x in X]
    seen, rep = set(q[:n_init]), 0
    for p in q[n_init:]:
        rep += p in seen
        seen.add(p)
    return rep / max(1, len(q) - n_init)


def load_rollouts(paths):
    rows = []
    for p in paths:
        for line in open(p):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:  # a line cut by a job time limit
                continue
            rows.append({k: d[k] for k in ("fid", "family", "seed", "tag", "symmetry", "boundary", "n_invalid")}
                        | {"p_nonnumber": d.get("p_nonnumber", np.nan), "lr": np.asarray(d["log_regret"], float),
                           "repeat": repeat_rate(d["X"], d["n_init"])}
                        | run_metrics(d["log_regret"]))
    R = pd.DataFrame(rows)
    # resumed shards may hold a run twice: keep the last
    return R.drop_duplicates(["fid", "seed", "tag", "symmetry"], keep="last").reset_index(drop=True)


def load_teachers(corpus, fids, seeds):
    specs = load_specs(corpus)
    rows = []
    for t in iter_trajectories(corpus, fields=("fid", "seed", "mid", "log_regret"), split="test"):
        if t["fid"] in fids and t["seed"] in seeds:
            s = specs[t["fid"]]
            rows.append({"fid": t["fid"], "family": s["kind"], "seed": t["seed"], "mid": t["mid"],
                         "teacher": teacher_label(t["mid"], s["kind"]), "boundary": s["boundary"],
                         "edge_dist": edge_distance(s["argmin"]), "lr": np.asarray(t["log_regret"], float)}
                        | run_metrics(t["log_regret"]))
    return pd.DataFrame(rows)


def paired(a, b, metric, rs=None):
    """Per-function mean of (a - b) over the shared (function, seed) pairs. Negative = a better. With rs (the RS
    teacher runs) also the per-function RS mean over the same pairs, for the gap-closed share."""
    m = a.merge(b, on=["fid", "seed"], suffixes=("_s", "_t"))
    if rs is not None:
        m = m.merge(rs[["fid", "seed", metric]].rename(columns={metric: f"{metric}_r"}), on=["fid", "seed"])
    m["d"] = m[f"{metric}_s"] - m[f"{metric}_t"]
    agg = dict(d=("d", "mean"), s=(f"{metric}_s", "mean"), t=(f"{metric}_t", "mean"),
               family=("family_s", "first"), boundary=("boundary_s", "first"))
    if rs is not None:
        agg["r"] = (f"{metric}_r", "mean")
    return m.groupby("fid").agg(**agg)


def groups(per_fn):
    out = [("all", per_fn)] + [(k, per_fn[per_fn.family == k]) for k in KINDS if (per_fn.family == k).any()]
    return out + [("boundary", per_fn[per_fn.boundary]), ("interior", per_fn[~per_fn.boundary])]


def summarize(per_fn):
    rows = []
    for name, g in groups(per_fn):
        if len(g) == 0:
            continue
        lo, hi = boot_ci(g.d.values)
        row = {"group": name, "n_fn": len(g), "student": g.s.mean(), "teacher": g.t.mean(), "diff": g.d.mean(),
               "ci_lo": lo, "ci_hi": hi}
        if "r" in g:
            den = g.t.mean() - g.r.mean()
            row["rs"] = g.r.mean()
            row["closed"] = (g.s.mean() - g.r.mean()) / den if abs(den) > 1e-9 else np.nan
        rows.append(row)
    if not rows:  # no shared (function, seed) pairs, e.g. incomplete rollouts
        return pd.DataFrame(columns=["n_fn", "student", "teacher", "diff", "ci_lo", "ci_hi"]).rename_axis("group")
    return pd.DataFrame(rows).set_index("group")


def rollout_tables(R, T):
    """{(tag, teacher, metric): table}. Every student tag vs the teacher it should imitate, plus oracle-tag vs the
    matched MLE teacher (the go/no-go comparison)."""
    R0 = R[R.symmetry == 0]
    rs = T[T.teacher == "rs"]
    out = {}
    tags = set(R0.tag)
    comps = [(tag, TAG_TEACHER[tag]) for tag in sorted(tags) if tag in TAG_TEACHER]
    if "GP-{K}-ORACLE" in tags:
        comps.append(("GP-{K}-ORACLE", "mle_matched"))
    if "UNK" in tags:
        comps += [("UNK", "oracle"), ("UNK", "mle_matched")]
    for tag, teacher in comps:
        s = R0[R0.tag == tag]
        t = T[T.teacher == teacher]
        for metric in ("auc", "auc_floor"):
            out[(tag, teacher, metric)] = summarize(paired(s, t, metric, rs if teacher != "rs" else None))
    return out


STUDENT_PAIRS = [("GP-{K}-ORACLE", "GP-{K}-MLE"), ("GP-{K}-ORACLE", "UNK"), ("GP-{K}-MLE", "UNK")]


def student_tables(R):
    """{(tag_a, tag_b, metric): table}: two prompts of the same model, paired per (function, seed)."""
    R0 = R[R.symmetry == 0]
    out = {}
    for a, b in STUDENT_PAIRS:
        if a in set(R0.tag) and b in set(R0.tag):
            for metric in ("auc", "auc_floor"):
                out[(a, b, metric)] = summarize(paired(R0[R0.tag == a], R0[R0.tag == b], metric))
    return out


# ---------------------------------------------------------------------------- regret vs trial
def mean_curve(df, floor=True):
    L = np.stack(df.lr.values)
    return (np.maximum(L, LF) if floor else L).mean(0)


def regret_curves(R, T):
    """Long table: group (all / family), who (student tag / teacher), trial (1-based), mean floored log regret.
    Only (function, seed) pairs present for the students are used for the teachers too."""
    R0 = R[(R.symmetry == 0) & (R.tag != "RS-baseline")]
    pairs = R0[["fid", "seed"]].drop_duplicates()
    T = T.merge(pairs, on=["fid", "seed"])
    rows = []
    for gname in ["all"] + KINDS:
        rg = R0 if gname == "all" else R0[R0.family == gname]
        tg = T if gname == "all" else T[T.family == gname]
        if len(rg) == 0:
            continue
        series = {f"student {tag}": rg[rg.tag == tag] for tag in sorted(set(rg.tag))}
        series |= {f"teacher {t}": tg[tg.teacher == t] for t in ("oracle", "mle_matched", "rs")}
        for who, df in series.items():
            if len(df):
                c = mean_curve(df)
                rows += [{"group": gname, "who": who, "trial": i + 1, "log_regret_floor": v} for i, v in enumerate(c)]
    return pd.DataFrame(rows)


def plot_regret_curves(C, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gs = [g for g in ["all"] + KINDS if g in set(C.group)]
    fig, axes = plt.subplots(1, len(gs), figsize=(4.2 * len(gs), 3.8), sharey=True, squeeze=False)
    for ax, g in zip(axes[0], gs):
        for who, d in C[C.group == g].groupby("who", sort=False):
            kind, name = who.split(" ", 1)
            color = TAG_COLOR.get(name, "#333333") if kind == "student" else TEACHER_COLOR.get(name, "#333333")
            ax.plot(d.trial, d.log_regret_floor, color=color, lw=2 if kind == "student" else 1.5,
                    ls="-" if kind == "student" else "--", label=who)
        ax.axhline(LF, color="0.5", ls=":", lw=1)
        ax.set_title(g)
        ax.set_xlabel("trial")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel("mean log10 regret (floored 1e-3)")
    axes[0][-1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{title}  (solid = student by prompt tag, dashed = teacher)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------- tag likelihood
def relative_tag(tag, family):
    if tag in ("RS", "UNK"):
        return tag
    if tag.startswith("GP-") and tag.count("-") == 2:
        _, k, mode = tag.split("-")
        k = k.lower()
        if mode == "ORACLE":
            return "ORACLE(matched)" if k == family else "ORACLE(other)"
        if mode == "MLE":
            return "MLE-matched" if k == family else f"MLE-{k}"
    return tag


def confusion(S, value="nll_x", stat="mean"):
    """Rows = teacher of the trace, columns = prompt tag. stat over the BO steps of a trace, then over traces."""
    S = S[~S.init].copy()
    S["row"] = [teacher_label(m, f) for m, f in zip(S.mid, S.family)]
    S["col"] = [relative_tag(t, f) for t, f in zip(S.tag, S.family)]
    per_trace = S.groupby(["fid", "seed", "mid", "row", "col"])[value].agg(stat).reset_index()
    M = per_trace.pivot_table(index="row", columns="col", values=value, aggfunc=stat)
    order_r = [r for r in ["oracle", "mle_matched", "mle|rbf", "mle|m12", "mle|m32", "mle|m52", "rs"] if r in M.index]
    order_c = [c for c in ["ORACLE(matched)", "ORACLE(other)", "MLE-matched", "MLE-rbf", "MLE-m12", "MLE-m32",
                           "MLE-m52", "RS", "UNK"] if c in M.columns]
    return M.loc[order_r, order_c]


def separation(M):
    """Does each teacher's trace get its lowest NLL under its own tag group? Groups: oracle -> ORACLE(matched),
    MLE rows -> any MLE column, rs -> RS. Also the exact (kernel-level) diagonal for the MLE rows."""
    mle_cols = [c for c in M.columns if c.startswith("MLE-")]
    res = {}
    if "oracle" in M.index:
        others = [c for c in M.columns if c not in ("ORACLE(matched)", "UNK")]
        res["oracle"] = bool(M.at["oracle", "ORACLE(matched)"] < M.loc["oracle", others].min())
    for r in [r for r in M.index if r.startswith("mle")]:
        own = M.loc[r, mle_cols].min()
        others = [c for c in ("ORACLE(matched)", "RS") if c in M.columns]
        res[r] = bool(own < M.loc[r, others].min())
        exact = "MLE-matched" if r == "mle_matched" else "MLE-" + r.split("|")[1]
        res[f"{r} exact kernel"] = bool(M.loc[r, mle_cols].idxmin() == exact)
    if "rs" in M.index:
        others = [c for c in M.columns if c not in ("RS", "UNK")]
        res["rs"] = bool(M.at["rs", "RS"] < M.loc["rs", others].min())
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--rollouts", nargs="*", default=[])
    ap.add_argument("--scores", nargs="*", default=[])
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    say, verdict = [], {}

    if args.rollouts:
        R = load_rollouts(args.rollouts)
        T = load_teachers(args.corpus, set(R.fid), set(R.seed))
        R.drop(columns="lr").to_csv(out / "rollout_runs.csv", index=False)
        say += [f"== 1. ROLLOUTS: {R.fid.nunique()} functions, seeds {sorted(map(int, R.seed.unique()))}, tags "
                f"{sorted(R.tag.unique())}; mean probability on non-number tokens at proposal positions (masked "
                f"out): {R.p_nonnumber.mean():.3f}",
                "runs per tag: " + ", ".join(f"{k} {v}" for k, v in R.groupby("tag").size().items()),
                "repeated proposals per tag (share of BO steps re-proposing an earlier point): "
                + ", ".join(f"{k} {v:.3f}" for k, v in R.groupby("tag").repeat.mean().items()),
                "diff = student - teacher (negative = student better), per-function mean over seeds, 95% bootstrap "
                "CI over functions; closed = share of the RS -> teacher gap the student closes"]
        for (tag, teacher, metric), tab in rollout_tables(R, T).items():
            if tab.empty:
                say += ["", f"-- student[{tag}] vs teacher {teacher}, {metric}: no shared runs"]
                continue
            tab.to_csv(out / f"rollout_{tag.replace('{K}', 'K')}_vs_{teacher}_{metric}.csv")
            say += ["", f"-- student[{tag}] vs teacher {teacher}, {metric}", tab.round(3).to_string()]
            a = tab.loc["all"]
            if tag == "GP-{K}-ORACLE" and teacher == "mle_matched" and metric == "auc_floor":
                verdict["oracle_tag_beats_matched_mle_floor"] = bool(a.ci_hi < 0)
                verdict["oracle_tag_vs_matched_mle_floor_diff"] = [round(a["diff"], 4), round(a.ci_lo, 4),
                                                                   round(a.ci_hi, 4)]
                verdict["oracle_tag_matches_mle_at_least"] = bool(a.ci_lo <= 0)
            if metric == "auc_floor" and "closed" in a and teacher == TAG_TEACHER.get(tag):
                verdict[f"gap_closed_floor {tag} vs {teacher}"] = round(float(a.closed), 3)
        say += ["", "== 1b. STUDENT vs STUDENT (same model, two prompts; diff = first - second)"]
        for (ta, tb, metric), tab in student_tables(R).items():
            if tab.empty:
                say += ["", f"-- student[{ta}] vs student[{tb}], {metric}: no shared (function, seed) pairs"]
                continue
            tab = tab.rename(columns={"student": ta, "teacher": tb})
            tab.to_csv(out / f"student_{ta.replace('{K}', 'K')}_vs_{tb.replace('{K}', 'K')}_{metric}.csv")
            say += ["", f"-- student[{ta}] vs student[{tb}], {metric}", tab.round(3).to_string()]
        C = regret_curves(R, T)
        C.to_csv(out / "regret_curves.csv", index=False)
        plot_regret_curves(C, out / "regret_curves.png", title=out.name)
        W = C[C.group == "all"].pivot_table(index="trial", columns="who", values="log_regret_floor")
        W = W.loc[[t for t in (5, 10, 20, 30, 40, 50) if t in W.index]]
        say += ["", "== 1c. REGRET vs TRIAL (all functions, mean floored log10 regret; full curves per family in "
                    "regret_curves.csv / .png)", W.round(3).to_string()]
        means = T.groupby("teacher").auc_floor.mean().rename("teacher_auc_floor")
        say += ["", "-- teacher means on these runs (auc_floor)", means.round(3).to_string()]

    if args.scores:
        S = pd.concat([pd.read_parquet(p) for p in args.scores], ignore_index=True)
        S["hit_0.02"] = (S.dist < 0.02).astype(float)
        S["hit_0.05"] = (S.dist < 0.05).astype(float)
        say += ["", f"== 2. TAG-LIKELIHOOD CONFUSION ({S.fid.nunique()} functions, BO steps only): rows = teacher "
                    f"of the trace, columns = prompt tag; entry = mean per-step NLL of the next x (nats, x0 + x1)"]
        M = confusion(S, "nll_x")
        M.to_csv(out / "confusion_nll_x.csv")
        sep = separation(M)
        say += [M.round(3).to_string()]
        views = [("dist", "mean", "next-x distance to the teacher, mean (greedy, teacher-forced, [0,1]^2 units)"),
                 ("dist", "median", "next-x distance to the teacher, median (robust to jumps between modes)"),
                 ("hit_0.02", "mean", "hit rate: greedy next x within 0.02 of the teacher's"),
                 ("hit_0.05", "mean", "hit rate: greedy next x within 0.05 of the teacher's")]
        for value, stat, text in views:
            V = confusion(S, value, stat)
            V.to_csv(out / f"confusion_{value}_{stat}.csv")
            say += ["", f"-- {text}", V.round(3).to_string()]
        say += ["", "-- separation (row minimum in its own tag group)", json.dumps(sep, indent=1)]
        core = {k: v for k, v in sep.items() if "exact" not in k}
        verdict["tag_likelihood_separates_oracle_mle_rs"] = bool(all(core.values()))

    say += ["", "== 3. GO / NO-GO", json.dumps(verdict, indent=1)]
    json.dump(verdict, open(out / "verdict.json", "w"), indent=1)
    text = "\n".join(say)
    (out / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
