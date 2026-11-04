"""
Paired comparison of several students ("arms": models, training data, decoding settings) on the same test runs.

Each arm is label=glob of its rollout files (run_gp_benchmark.py output). The first arm is the reference. For every
prompt tag that two arms share, runs are paired per (function, seed); rollout seeds do not depend on the arm, so the
same (function, seed) starts from the same initial design in every arm.

label@TAG=glob uses only the runs prompted with TAG and files them under the tag "chosen": arms with different
prompts (e.g. BEST_only@BEST, ALL@UNK, BEST_fixed@GP-M52-MLE) are then paired directly, each against the selection
baselines (teacher sel_loso, closed = share of the sel_fixed -> sel_loso gap). Q1 / Q2 use this.

    python benchmarks/gp_functions/compare_gp_arms.py --corpus corpus \
        --arms 'R=rollouts/R_shard_*.jsonl' 'R_greedy=rollouts/R_greedy_shard_*.jsonl' --output_path compare/

Output (--output_path):
    summary.csv / report.txt   per tag and arm, on the (function, seed) pairs ALL arms with that tag share:
                               auc_floor, auc, share of the RS -> teacher gap closed, diff vs the reference [95% CI],
                               repeat (share of proposals re-proposing an earlier point)
    <tag>_<arm>_vs_<ref>_<metric>.csv   the full paired table (all / family / boundary / interior), on the pairs
                               the two arms share
    curves_<tag>.png           regret vs trial per family: every arm, the teacher it imitates and RS
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_gp_eval import (KINDS, LF, add_selection_baselines, boot_ci, load_rollouts, load_teachers,  # noqa: E402
                             mean_curve, paired, summarize, tag_base, tag_reference, teacher_rows)

# fixed categorical order for arms (never cycled: more arms than colours is an error); adjacent pairs checked for
# colour-vision-deficiency separation; every arm is also named in the legend
ARM_COLORS = ["#d62728", "#1f77b4", "#e08a00", "#7b4fbf", "#2ca02c", "#d04f9e", "#6b6b00", "#0f9fb5"]
TAGS = ["BEST", "GP-{K}-ORACLE", "GP-{K}-MLE", "UNK", "RS"]  # order of the report; other tags (literal) follow


def parse_arms(specs):
    arms = {}
    for s in specs:
        label, _, pattern = s.partition("=")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[WARN] arm {label}: no files match {pattern}, skipped")
            continue
        arms[label] = files
    return arms


def teacher_for(tag):
    return tag_reference(tag) or "oracle"  # UNK is measured against the oracle, BEST against sel_loso


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="label=glob; the first arm is the reference")
    ap.add_argument("--tags", nargs="+", default=None, help="default: every tag of any arm")
    ap.add_argument("--ranks", default=None, help="teacher_ranks.parquet (best fixed teacher chosen on train)")
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 30)

    arms = parse_arms(args.arms)
    if len(arms) > len(ARM_COLORS):
        raise SystemExit(f"at most {len(ARM_COLORS)} arms per comparison (got {len(arms)})")
    R = {}
    for spec, files in arms.items():
        label, _, chosen = spec.partition("@")
        r = load_rollouts(files)
        r = r[r.symmetry == 0]
        if chosen:
            if not (r.tag == chosen).any():
                print(f"[WARN] arm {label}: no runs with tag {chosen}, skipped")
                continue
            r = r[r.tag == chosen].assign(tag="chosen")
            label = f"{label}[{chosen}]"
        R[label] = r
    ref = next(iter(R))
    fids = set().union(*(set(r.fid) for r in R.values()))
    seeds = set().union(*(set(r.seed) for r in R.values()))
    T, sel_info = add_selection_baselines(load_teachers(args.corpus, fids, seeds), args.ranks)
    present = set().union(*(set(r.tag) for r in R.values()))
    tags = args.tags or [t for t in TAGS if t in present] + sorted(present - set(TAGS))

    say = [f"== ARM COMPARISON, reference = {ref}", "arms: " + ", ".join(f"{a} ({len(r)} runs)" for a, r in R.items()),
           "closed = share of the RS -> teacher gap closed (1 = teacher, 0 = RS); diff = arm - reference "
           "(negative = arm better), per-function mean over seeds, 95% bootstrap CI over functions; for BEST the "
           "teacher is sel_loso (per-function best on the other seeds) and closed is the share of the sel_fixed -> "
           "sel_loso gap",
           f"best fixed teacher: {sel_info.get('fixed')} (chosen on {sel_info.get('fixed_chosen_on')})"]
    summary = []
    for tag in tags:
        have = {a: r[r.tag == tag] for a, r in R.items() if (r.tag == tag).any()}
        if not have:
            continue
        teacher = teacher_rows(T, teacher_for(tag))
        rs = T[T.teacher == tag_base(tag)]
        # common (function, seed) pairs of all arms with this tag: the summary compares like with like
        common = None
        for r in have.values():
            p = set(zip(r.fid, r.seed))
            common = p if common is None else common & p
        say += ["", f"-- tag {tag} (teacher {teacher_for(tag)}), {len(common)} common (function, seed) pairs, "
                    f"{len({f for f, _ in common})} functions"]
        keep = pd.DataFrame(sorted(common), columns=["fid", "seed"])
        if not common:
            say += ["   no (function, seed) pair is shared by all arms: summary skipped"]
        for a, r in have.items():
            rc = r.merge(keep, on=["fid", "seed"])
            row = {"tag": tag, "arm": a, "n_fn": rc.fid.nunique(), "repeat": rc.repeat.mean()}
            for metric in ("auc_floor", "auc"):
                st = summarize(paired(rc, teacher, metric, rs))
                if st.empty:
                    continue
                pt = st.loc["all"]
                row[metric] = pt.student
                row[f"closed_{metric}"] = pt.closed
                if a != ref and ref in have:
                    pr = paired(rc, have[ref].merge(keep, on=["fid", "seed"]), metric)
                    lo, hi = boot_ci(pr.d.values)
                    row[f"diff_{metric}"], row[f"ci_lo_{metric}"], row[f"ci_hi_{metric}"] = pr.d.mean(), lo, hi
            summary.append(row)
            if a != ref and ref in have:
                # full tables on the pairs these two arms share (more functions than the all-arm intersection)
                for metric in ("auc_floor", "auc"):
                    tab = summarize(paired(r, have[ref], metric)).rename(columns={"student": a, "teacher": ref})
                    if tab.empty:
                        continue
                    tab.to_csv(out / f"{tag.replace('{K}', 'K')}_{a}_vs_{ref}_{metric}.csv")
                    if metric == "auc_floor":
                        say += ["", f"   {a} vs {ref}, {metric} (on their {int(tab.loc['all'].n_fn)} shared functions)",
                                tab.round(3).to_string()]
        S = pd.DataFrame([s for s in summary if s["tag"] == tag]).set_index("arm").drop(columns="tag")
        say += ["", "   summary on the common pairs", S.round(3).to_string()]
        plot_curves(have, teacher, rs, keep, tag, out / f"curves_{tag.replace('{K}', 'K')}.png")

    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    text = "\n".join(say)
    (out / "report.txt").write_text(text)
    print(text)


def plot_curves(have, teacher, rs, keep, tag, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = ["all"] + KINDS
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.8), sharey=True)
    for ax, g in zip(axes, groups):
        sel = (lambda d: d) if g == "all" else (lambda d, g=g: d[d.family == g])
        drawn = False
        for (a, r), c in zip(have.items(), ARM_COLORS):
            d = sel(r.merge(keep, on=["fid", "seed"]))
            if len(d):
                ax.plot(np.arange(1, 51), mean_curve(d), color=c, lw=2, label=a)
                drawn = True
        if not drawn:
            ax.set_visible(False)
            continue
        for d, ls in ((teacher, "--"), (rs, ":")):
            if d.empty:
                continue
            name = f"teacher {d.teacher.iloc[0]}"
            d = sel(d.merge(keep, on=["fid", "seed"]))
            if len(d):
                ax.plot(np.arange(1, 51), mean_curve(d), color="#222222", lw=1.5, ls=ls, label=name)
        ax.axhline(LF, color="0.6", ls=":", lw=0.8)
        ax.set_title(g)
        ax.set_xlabel("trial")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("mean log10 regret (floored 1e-3)")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"student prompted {tag}: arms vs teacher (common runs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
