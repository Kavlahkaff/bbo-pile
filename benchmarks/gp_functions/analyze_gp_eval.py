"""
Analysis of student rollouts (run_gp_benchmark.py) and teacher-forced scores (score_teacher_traces.py).

Metrics: log10 normalised simple regret (fmin / fmax of the spec), AUC = mean over the 50 trials, raw and floored at
1e-3. Comparisons are paired per (function, seed), averaged per function, with bootstrap CIs over functions; every
table is given for all functions, per family, and split boundary / interior.

    python benchmarks/gp_functions/analyze_gp_eval.py --corpus corpus --rollouts rollouts.jsonl \
        --scores scores.parquet --output_path eval/

Sections
    1. student (per prompt tag) vs each teacher it should imitate + the go/no-go comparison
       student prompted as oracle vs the matched-kernel MLE teacher (floored)
    2. tag-likelihood confusion matrix: rows = teacher of the trace, columns = prompt tag (relative to the function
       family), entry = mean per-step NLL of the teacher's next x over the BO steps; plus the next-x distance view
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
from open_optformer.gp_corpus import FLOOR, edge_distance, iter_trajectories, load_specs  # noqa: E402

LF = math.log10(FLOOR)

# which teacher a prompt tag template should imitate (kernel_mode scheme)
TAG_TEACHER = {"GP-{K}-ORACLE": "oracle", "GP-{K}-MLE": "mle_matched", "RS": "rs", "RS-baseline": "rs"}


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


def load_rollouts(paths):
    rows = []
    for p in paths:
        for line in open(p):
            d = json.loads(line)
            rows.append({k: d[k] for k in ("fid", "family", "seed", "tag", "symmetry", "boundary", "n_invalid")}
                        | run_metrics(d["log_regret"]))
    return pd.DataFrame(rows)


def load_teachers(corpus, fids, seeds):
    specs = load_specs(corpus)
    rows = []
    for t in iter_trajectories(corpus, fields=("fid", "seed", "mid", "log_regret")):
        if t["fid"] in fids and t["seed"] in seeds:
            s = specs[t["fid"]]
            rows.append({"fid": t["fid"], "family": s["kind"], "seed": t["seed"], "mid": t["mid"],
                         "teacher": teacher_label(t["mid"], s["kind"]), "boundary": s["boundary"],
                         "edge_dist": edge_distance(s["argmin"])} | run_metrics(t["log_regret"]))
    return pd.DataFrame(rows)


def paired(student, teacher, metric):
    """Per-function mean of (student - teacher) over the shared seeds. Negative = student better."""
    m = student.merge(teacher, on=["fid", "seed"], suffixes=("_s", "_t"))
    m["d"] = m[f"{metric}_s"] - m[f"{metric}_t"]
    per_fn = m.groupby("fid").agg(d=("d", "mean"), s=(f"{metric}_s", "mean"), t=(f"{metric}_t", "mean"),
                                  family=("family_s", "first"), boundary=("boundary_s", "first"))
    return per_fn


def summarize(per_fn):
    rows = []
    groups = [("all", per_fn)] + list(per_fn.groupby("family")) + \
             [("boundary", per_fn[per_fn.boundary]), ("interior", per_fn[~per_fn.boundary])]
    for name, g in groups:
        if len(g) == 0:
            continue
        lo, hi = boot_ci(g.d.values)
        rows.append({"group": name, "n_fn": len(g), "student": g.s.mean(), "teacher": g.t.mean(), "diff": g.d.mean(),
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).set_index("group")


def rollout_tables(R, T):
    """{(tag, teacher, metric): table}. Every student tag vs the teacher it should imitate, plus oracle-tag vs the
    matched MLE teacher (the go/no-go comparison)."""
    R0 = R[R.symmetry == 0]
    out = {}
    comps = [(tag, TAG_TEACHER[tag]) for tag in R0.tag.unique() if tag in TAG_TEACHER]
    if "GP-{K}-ORACLE" in set(R0.tag):
        comps.append(("GP-{K}-ORACLE", "mle_matched"))
    if "UNK" in set(R0.tag):
        comps += [("UNK", "oracle"), ("UNK", "mle_matched")]
    for tag, teacher in comps:
        s = R0[R0.tag == tag]
        t = T[T.teacher == teacher]
        for metric in ("auc", "auc_floor"):
            out[(tag, teacher, metric)] = summarize(paired(s, t, metric))
    return out


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


def confusion(S, value="nll_x"):
    S = S[~S.init].copy()
    S["row"] = [teacher_label(m, f) for m, f in zip(S.mid, S.family)]
    S["col"] = [relative_tag(t, f) for t, f in zip(S.tag, S.family)]
    per_trace = S.groupby(["fid", "seed", "mid", "row", "col"])[value].mean().reset_index()
    M = per_trace.pivot_table(index="row", columns="col", values=value, aggfunc="mean")
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
    pd.set_option("display.width", 200, "display.max_columns", 30)
    say, verdict = [], {}

    if args.rollouts:
        R = load_rollouts(args.rollouts)
        T = load_teachers(args.corpus, set(R.fid), set(R.seed))
        R.to_csv(out / "rollout_runs.csv", index=False)
        say += [f"== 1. ROLLOUTS: {R.fid.nunique()} functions, seeds {sorted(R.seed.unique())}, tags "
                f"{sorted(R.tag.unique())}; invalid proposals per run: {R.n_invalid.mean():.2f}",
                "diff = student - teacher (negative = student better), per-function mean over seeds, 95% bootstrap "
                "CI over functions"]
        for (tag, teacher, metric), tab in rollout_tables(R, T).items():
            tab.to_csv(out / f"rollout_{tag.replace('{K}', 'K')}_vs_{teacher}_{metric}.csv")
            say += ["", f"-- student[{tag}] vs teacher {teacher}, {metric}", tab.round(3).to_string()]
            if tag == "GP-{K}-ORACLE" and teacher == "mle_matched" and metric == "auc_floor":
                a = tab.loc["all"]
                verdict["oracle_tag_beats_matched_mle_floor"] = bool(a.ci_hi < 0)
                verdict["oracle_tag_vs_matched_mle_floor_diff"] = [round(a["diff"], 4), round(a.ci_lo, 4),
                                                                   round(a.ci_hi, 4)]
                verdict["oracle_tag_matches_mle_at_least"] = bool(a.ci_lo <= 0)
        means = T.groupby("teacher").auc_floor.mean().rename("teacher_auc_floor")
        say += ["", "-- teacher means on these runs (auc_floor)", means.round(3).to_string()]

    if args.scores:
        S = pd.concat([pd.read_parquet(p) for p in args.scores], ignore_index=True)
        say += ["", f"== 2. TAG-LIKELIHOOD CONFUSION ({S.fid.nunique()} functions, BO steps only): rows = teacher "
                    f"of the trace, columns = prompt tag; entry = mean per-step NLL of the next x (nats, x0 + x1)"]
        M = confusion(S, "nll_x")
        M.to_csv(out / "confusion_nll_x.csv")
        D = confusion(S, "dist")
        D.to_csv(out / "confusion_dist.csv")
        sep = separation(M)
        say += [M.round(3).to_string(), "", "-- next-x distance to the teacher (greedy, teacher-forced, [0,1]^2 units)",
                D.round(3).to_string(), "", "-- separation (row minimum in its own tag group)",
                json.dumps(sep, indent=1)]
        core = {k: v for k, v in sep.items() if "exact" not in k}
        verdict["tag_likelihood_separates_oracle_mle_rs"] = bool(all(core.values()))

    if verdict:
        say += ["", "== 3. GO / NO-GO", json.dumps(verdict, indent=1)]
        json.dump(verdict, open(out / "verdict.json", "w"), indent=1)
    text = "\n".join(say)
    (out / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
