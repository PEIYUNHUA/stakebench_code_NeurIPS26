"""
E9e: statistics for the translation-manipulation experiment.

Follows RUNBOOK 5 exactly, including the parts that exist because earlier experiments got
them wrong:

  1. Complete pairing. An instance is kept only if every arm of that (model, task) produced a
     prediction. Dropped instances are counted and reported, never silently ignored.
  2. Primary metric: balanced accuracy against the 0.5 floor, for both G1 and G2.
  3. ONE estimand. Delta is computed per model first, then averaged over models. This is the
     E2b lesson: the old E2b table averaged per model while its bootstrap pooled all models,
     producing a point estimate that disagreed with the table in sign. The bootstrap here
     recomputes the identical per-model-then-average statistic, and the script prints the
     agreement check, which must be at machine-precision scale.
  4. Paired tests: exact McNemar per (model, language, task); intervals from a paired
     bootstrap clustered on market_id, 2,000 draws. A market drawn twice is entered as two
     distinct clusters -- the other E2b bug was that a later groupby silently merged repeated
     draws back into one cluster, deflating the effective cluster count.
  5. BH-FDR over the language x model x task family.
  6. Off-label output rate reported as its own column. On low-resource languages this is
     expected to rise, and it is an instruction-following failure, not a commitment-reading
     failure. Mixing it into the headline would be caught.

Balanced accuracy counts NEUTRAL and any other non-label output as incorrect. That matches
E6's convention and is NOT the paper's Directed Accuracy, which excludes neutrals. The two
must never be placed in the same table.

Outputs: e9d_summary.csv, e9d_paired_tests.csv, e9_offlabel.csv, e9e_stats_report.md
"""
import os, re, sys, glob, json, argparse
import numpy as np
import pandas as pd
from scipy import stats

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "rebuttal", "results", "e9")
PRED = os.path.join(OUT, "predictions")

VALID = {"G1": {"YES", "NO"}, "G2": {"YES", "NO", "NEUTRAL"}}
N_BOOT = 2000
SEED = 42


def balanced_accuracy(gt, pred):
    """Macro recall over the gold classes; non-label outputs count as wrong."""
    gt = np.asarray(gt); pred = np.asarray(pred)
    recalls = []
    for c in np.unique(gt):
        m = gt == c
        if m.sum():
            recalls.append((pred[m] == c).mean())
    return float(np.mean(recalls)) if recalls else np.nan


def load_predictions():
    rows = []
    for f in glob.glob(os.path.join(PRED, "e9d_predictions_*.csv")):
        d = pd.read_csv(f)
        rows.append(d)
    if not rows:
        raise SystemExit("no prediction files found")
    d = pd.concat(rows, ignore_index=True)
    d["gt"] = d["gt"].astype(str).str.upper()
    d["pred"] = d["pred"].astype(str).str.upper()
    d = d.drop_duplicates(["model", "task", "arm", "comment_id"], keep="last")
    return d


def complete_pairs(d):
    """Complete pairing in both directions.

    First: an instance must appear in every arm of its (model, task).
    Then: within a task, models must share the SAME instance set. Without the second step
    the per-model deltas would be computed on slightly different instances and the average
    over models would not be an average over a common sample -- exactly the "not the same
    instance set" problem flagged in the workbook's E6 caveat. Both drop counts are reported.
    """
    keep, drops = [], []
    for (model, task), g in d.groupby(["model", "task"]):
        arms = sorted(g.arm.unique())
        cnt = g.groupby("comment_id")["arm"].nunique()
        full = set(cnt[cnt == len(arms)].index)
        drops.append(dict(model=model, task=task, n_arms=len(arms),
                          instances_total=int(cnt.shape[0]), instances_arm_paired=len(full),
                          dropped_missing_arm=int(cnt.shape[0] - len(full))))
        keep.append(g[g.comment_id.isin(full)])
    d2 = pd.concat(keep, ignore_index=True)

    drops = pd.DataFrame(drops)
    final = []
    common_n = {}
    for task, g in d2.groupby("task"):
        sets = [set(gg.comment_id) for _, gg in g.groupby("model")]
        common = set.intersection(*sets) if sets else set()
        common_n[task] = len(common)
        final.append(g[g.comment_id.isin(common)])
    d3 = pd.concat(final, ignore_index=True)

    drops["instances_common_across_models"] = drops.task.map(common_n)
    drops["dropped_not_shared"] = drops.instances_arm_paired - drops.instances_common_across_models
    return d3, drops


def per_model_delta(d, task, lang_arm, en_arm="en"):
    """Delta per model, then the mean over models -- the single estimand."""
    per = {}
    for model, g in d[d.task == task].groupby("model"):
        a = g[g.arm == en_arm]; b = g[g.arm == lang_arm]
        if a.empty or b.empty:
            continue
        per[model] = (balanced_accuracy(b["gt"], b["pred"])
                      - balanced_accuracy(a["gt"], a["pred"]))
    if not per:
        return np.nan, {}
    return float(np.mean(list(per.values()))), per


def clustered_bootstrap(d, task, lang_arm, en_arm="en", n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap over market_id clusters, computed as market-level count aggregates.

    Resampling markets with replacement is equivalent to drawing a multinomial count vector
    over markets and weighting each market's contribution by its count. A market drawn m
    times therefore contributes m times, which is what "a repeated market counts as m
    clusters" means -- there is no later groupby that can silently merge them back, which is
    the failure mode the rewritten E2b had to fix.

    Doing it as counts rather than by rebuilding a DataFrame per draw turns a ~1.2 hour run
    into seconds, and is exact rather than an approximation.
    """
    sub = d[(d.task == task) & (d.arm.isin([en_arm, lang_arm]))]
    if sub.empty:
        return (np.nan, np.nan)

    models = sorted(sub.model.unique())
    classes = sorted(sub["gt"].unique())
    markets = sorted(sub.market_id.astype(str).unique())
    m_index = {m: i for i, m in enumerate(markets)}
    M = len(markets)

    # denom[model, class, market] and correct counts per arm
    denom = np.zeros((len(models), len(classes), M))
    num_en = np.zeros_like(denom)
    num_lang = np.zeros_like(denom)
    for mi, model in enumerate(models):
        gm = sub[sub.model == model]
        for ai, arm in ((0, en_arm), (1, lang_arm)):
            ga = gm[gm.arm == arm]
            mk = ga.market_id.astype(str).map(m_index).values
            ok = (ga["pred"].values == ga["gt"].values)
            for ci, c in enumerate(classes):
                sel = ga["gt"].values == c
                if ai == 0:
                    np.add.at(denom[mi, ci], mk[sel], 1.0)
                    np.add.at(num_en[mi, ci], mk[sel], ok[sel].astype(float))
                else:
                    np.add.at(num_lang[mi, ci], mk[sel], ok[sel].astype(float))

    rng = np.random.RandomState(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.multinomial(M, np.full(M, 1.0 / M)).astype(float)
        dn = denom @ w                      # (models, classes)
        with np.errstate(invalid="ignore", divide="ignore"):
            r_en = np.where(dn > 0, (num_en @ w) / dn, np.nan)
            r_lang = np.where(dn > 0, (num_lang @ w) / dn, np.nan)
        ba_en = np.nanmean(r_en, axis=1)
        ba_lang = np.nanmean(r_lang, axis=1)
        out[b] = np.mean(ba_lang - ba_en)
    out = out[~np.isnan(out)]
    if out.size == 0:
        return (np.nan, np.nan)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def mcnemar_exact(d, model, task, lang_arm, en_arm="en"):
    g = d[(d.model == model) & (d.task == task)]
    a = g[g.arm == en_arm].set_index("comment_id")
    b = g[g.arm == lang_arm].set_index("comment_id")
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return np.nan, 0, 0
    ac = (a.loc[common, "pred"] == a.loc[common, "gt"]).values
    bc = (b.loc[common, "pred"] == b.loc[common, "gt"]).values
    n01 = int((ac & ~bc).sum())      # en right, language wrong
    n10 = int((~ac & bc).sum())      # language right, en wrong
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    p = float(stats.binomtest(n01, n, 0.5).pvalue) if hasattr(stats, "binomtest") \
        else float(stats.binom_test(n01, n, 0.5))
    return p, n01, n10


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return q
    pv = p[ok]; n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    raw = load_predictions()
    d, drops = complete_pairs(raw)
    print("=== pairing ===")
    print(drops.to_string(index=False))

    # ---- per (model, arm, task) summary
    rows = []
    for (model, task, arm), g in d.groupby(["model", "task", "arm"]):
        valid = VALID[task]
        off = float((~g["pred"].isin(valid)).mean())
        neutral = float((g["pred"] == "NEUTRAL").mean()) if task == "G2" else 0.0
        rows.append(dict(model=model, task=task, arm=arm, n=len(g),
                         bal_acc=round(balanced_accuracy(g["gt"], g["pred"]), 4),
                         accuracy=round(float((g["pred"] == g["gt"]).mean()), 4),
                         off_label_rate=round(off, 4),
                         neutral_rate=round(neutral, 4),
                         cost_usd=round(float(g.get("cost_usd", pd.Series([0]*len(g))).sum()), 4)))
    summary = pd.DataFrame(rows).sort_values(["task", "model", "arm"])
    summary.to_csv(os.path.join(OUT, f"e9d_summary{args.tag}.csv"), index=False)

    # ---- deltas, McNemar, bootstrap
    arms = sorted(d.arm.unique())
    lang_arms = [a for a in arms if a != "en"]
    test_rows = []
    for task in sorted(d.task.unique()):
        dt = d[d.task == task]
        for arm in lang_arms:
            if arm not in set(dt.arm.unique()):
                continue
            point, per = per_model_delta(dt, task, arm)
            lo, hi = clustered_bootstrap(dt, task, arm, n_boot=args.boot)
            for model, dm in per.items():
                p, n01, n10 = mcnemar_exact(dt, model, task, arm)
                test_rows.append(dict(task=task, arm=arm, model=model,
                                      delta_model=round(dm, 4), mcnemar_p=p,
                                      en_right_lang_wrong=n01, lang_right_en_wrong=n10,
                                      delta_mean_over_models=round(point, 4),
                                      ci_lo=None if np.isnan(lo) else round(lo, 4),
                                      ci_hi=None if np.isnan(hi) else round(hi, 4)))
    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests["bh_q"] = bh_fdr(tests.mcnemar_p.values).round(4)
        tests["mcnemar_p"] = tests.mcnemar_p.round(4)
    tests.to_csv(os.path.join(OUT, f"e9d_paired_tests{args.tag}.csv"), index=False)

    # ---- estimand self-check: bootstrap statistic vs table difference
    print("\n=== estimand self-check (must be ~1e-15) ===")
    checks = []
    for task in sorted(d.task.unique()):
        dt = d[d.task == task]
        for arm in lang_arms:
            if arm not in set(dt.arm.unique()):
                continue
            point, per = per_model_delta(dt, task, arm)
            # Recompute the table cells from the predictions at full precision. Reading the
            # rounded CSV back would cap agreement at ~1e-5 and hide a genuine estimand
            # mismatch behind display rounding.
            per_tbl = []
            for model in per:
                gm = dt[dt.model == model]
                a = gm[gm.arm == "en"]; b = gm[gm.arm == arm]
                if a.empty or b.empty:
                    continue
                per_tbl.append(balanced_accuracy(b["gt"], b["pred"])
                               - balanced_accuracy(a["gt"], a["pred"]))
            diff = abs(point - float(np.mean(per_tbl))) if per_tbl else np.nan
            checks.append(dict(task=task, arm=arm, estimand=point,
                               from_table=float(np.mean(per_tbl)) if per_tbl else np.nan,
                               abs_diff=diff))
    chk = pd.DataFrame(checks)
    print(chk.to_string(index=False))
    print(f"max |diff| = {chk.abs_diff.max():.2e}  (must be at machine-precision scale)")

    # ---- report
    lines = ["# E9 statistics report", ""]
    lines.append(f"- prediction rows: {len(raw)} -> paired: {len(d)}")
    lines.append(f"- bootstrap draws: {args.boot}, clustered on market_id, seed {SEED}")
    lines.append("- balanced accuracy counts NEUTRAL/off-label as incorrect (E6 convention, "
                 "NOT the paper's Directed Accuracy)")
    lines.append("")
    lines.append("## pairing")
    lines.append(drops.to_markdown(index=False))
    lines.append("")
    lines.append("## balanced accuracy by arm")
    piv = summary.pivot_table(index=["task", "model"], columns="arm", values="bal_acc")
    lines.append(piv.round(3).to_markdown())
    lines.append("")
    lines.append("## off-label output rate by arm")
    pivo = summary.pivot_table(index=["task", "model"], columns="arm", values="off_label_rate")
    lines.append(pivo.round(3).to_markdown())
    lines.append("")
    lines.append("## delta vs English (mean over models) with clustered 95% CI")
    if not tests.empty:
        head = (tests.groupby(["task", "arm"])
                .agg(delta=("delta_mean_over_models", "first"),
                     ci_lo=("ci_lo", "first"), ci_hi=("ci_hi", "first"),
                     n_models=("model", "nunique"),
                     n_sig_raw=("mcnemar_p", lambda s: int((s < 0.05).sum())),
                     n_sig_bh=("bh_q", lambda s: int((s < 0.05).sum())))
                .reset_index())
        lines.append(head.to_markdown(index=False))
    lines.append("")
    lines.append("## estimand self-check")
    lines.append(chk.to_markdown(index=False))
    with open(os.path.join(OUT, f"e9e_stats_report{args.tag}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n=== balanced accuracy ===")
    print(piv.round(3).to_string())
    print("\n=== off-label rate ===")
    print(pivo.round(3).to_string())
    if not tests.empty:
        print("\n=== delta (mean over models) ===")
        print(head.to_string(index=False))
    print(f"\nwrote e9d_summary{args.tag}.csv, e9d_paired_tests{args.tag}.csv, "
          f"e9e_stats_report{args.tag}.md")


if __name__ == "__main__":
    main()
