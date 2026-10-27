"""
E9g: assemble e9_summary.md and e9h_examples.md.

The summary is written for a human reader and is deliberately conservative. RUNBOOK 9 fixes
the strongest sentence the design can support, and the three boundary statements in RUNBOOK 8
are mandatory. This script emits both, plus the result-to-wording mapping, so the wording is
generated from the numbers rather than chosen after looking at them.

The one sentence that must never appear -- "StakeBench transfers to non-English markets" --
is checked for on the way out.

Outputs: e9_summary.md, e9h_examples.md
"""
import os, sys, glob, json, argparse
import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "rebuttal", "results", "e9")
PRED = os.path.join(OUT, "predictions")

BOUNDARIES = [
    "Machine-translated text is not authentic non-English prediction-market discourse. This "
    "measures robustness to the surface language of translated text, not transfer to real "
    "non-English markets.",
    "G4 was not evaluated.",
    "The quality filter (`is_english`) and the English commitment lexicon were not validated "
    "in these languages; this experiment holds them fixed rather than re-deriving them.",
]

FORBIDDEN = ["transfers to non-english markets", "stakebench transfers"]


# A confidence interval that merely contains zero is NOT evidence of robustness. Calling a
# delta of -0.11 with CI [-0.24, +0.00] "robust" would be an overclaim of exactly the kind the
# workbook blacklists. "Robust" is reserved for intervals that actually exclude a meaningful
# effect, i.e. an equivalence-style conclusion; everything else that spans zero is reported as
# inconclusive and underpowered.
EQUIV_MARGIN = 0.05


def wording(delta, lo, hi, rt_delta):
    """RUNBOOK 9 result -> permitted conclusion."""
    have_ci = not (lo is None or hi is None or np.isnan(lo) or np.isnan(hi))
    sig = have_ci and (lo > 0 or hi < 0)
    if not sig:
        if have_ci and lo > -EQUIV_MARGIN and hi < EQUIV_MARGIN:
            return ("no change, interval tight",
                    f"The interval excludes changes larger than {EQUIV_MARGIN:.2f} in either "
                    "direction, so commitment reading is robust to the surface language of the "
                    "translated comment on this arm. Still not a transfer claim.")
        return ("inconclusive (interval spans zero and is wide)",
                "Underpowered on this arm: the interval contains zero but is too wide to rule "
                "out a substantial change. Report as inconclusive; do NOT describe this as "
                "robustness.")
    if delta < 0:
        recovered = rt_delta is not None and not np.isnan(rt_delta) and abs(rt_delta) < abs(delta) / 2
        if recovered:
            return ("drop, back-translation recovers",
                    "The gap comes from the surface language rather than lost content, since "
                    "back-translating the same text restores most of the performance.")
        return ("drop, back-translation also drops",
                "The gap is largely translation loss. This experiment cannot separate a "
                "language effect here, and that must be stated plainly.")
    return ("increase", "The language arm scores higher than English on this arm; report as "
                        "observed without constructing an explanation for it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t = args.tag

    summary = pd.read_csv(os.path.join(OUT, f"e9d_summary{t}.csv"))
    tests = pd.read_csv(os.path.join(OUT, f"e9d_paired_tests{t}.csv"))
    fid_path = os.path.join(OUT, f"e9a_translation_fidelity{t}.csv")
    fid = pd.read_csv(fid_path) if os.path.exists(fid_path) else pd.DataFrame()
    cc_path = os.path.join(OUT, f"e9_control_consistency{t}.csv")
    cc = pd.read_csv(cc_path) if os.path.exists(cc_path) else pd.DataFrame()
    prereg = {}
    pp = os.path.join(OUT, "e9_preregistration.json")
    if os.path.exists(pp):
        with open(pp, encoding="utf-8") as f:
            prereg = json.load(f)

    L = ["# E9 -- commitment reading under a controlled translation manipulation", ""]
    if prereg:
        L += [f"- pre-registration: `{prereg.get('git_hash','')[:10]}`, "
              f"instance list `{prereg.get('instance_list_sha256','')[:16]}`, "
              f"seed {prereg.get('inference',{}).get('seed')}", ""]
    L += ["Same instances, same gold, same prompt, same pipeline. The only thing that changes "
          "is the language the comment is written in. Labels come from market records, so "
          "translation cannot change any gold label -- every instance is perfectly paired "
          "across arms.", ""]

    L += ["## Translation fidelity (gate)", ""]
    if not fid.empty:
        L.append(fid.drop(columns=[c for c in ("notes",) if c in fid.columns])
                 .to_markdown(index=False))
        bad = fid[(fid.numeral_preservation.notna()) & (fid.numeral_preservation < 0.90)]
        L += ["", f"Numeral preservation gate (>= 0.90): "
                  f"{'FAILED for ' + ', '.join(bad.lang) if len(bad) else 'passed for all languages'}."]
        L += ["", "`human_ok_rate` is intentionally blank. RUNBOOK 3.4 requires a real human "
                  "judgement on 30 items per language; it is never machine-filled. See "
                  "`e9a_human_spotcheck_TOFILL.csv`.", ""]

    L += ["## Balanced accuracy by arm", ""]
    piv = summary.pivot_table(index=["task", "model"], columns="arm", values="bal_acc")
    L += [piv.round(3).to_markdown(), ""]
    L += ["Balanced accuracy counts NEUTRAL and any non-label output as incorrect. This is the "
          "E6 convention and is NOT the paper's Directed Accuracy, which excludes neutrals. "
          "Do not place the two in one table.", ""]

    L += ["## Change vs English", ""]
    head = (tests.groupby(["task", "arm"])
            .agg(delta=("delta_mean_over_models", "first"), ci_lo=("ci_lo", "first"),
                 ci_hi=("ci_hi", "first"), n_models=("model", "nunique"),
                 n_sig_bh=("bh_q", lambda s: int((s < 0.05).sum())))
            .reset_index())
    L += [head.round(4).to_markdown(index=False), ""]
    L += ["Estimand: per-model delta first, then the mean over models. The bootstrap "
          "recomputes that same statistic, clustered on `market_id`, and the self-check in "
          "`e9e_stats_report.md` confirms the point estimate equals the table difference.", ""]

    L += ["## Permitted wording, per arm", ""]
    rt_map = {(r.task, r.arm): r.delta for r in head.itertuples()}
    for r in head.itertuples():
        if r.arm.endswith("_rt"):
            continue
        rt = rt_map.get((r.task, r.arm + "_rt"), np.nan)
        tag, sentence = wording(r.delta, r.ci_lo, r.ci_hi, rt)
        L.append(f"- **{r.task} / {r.arm}** ({tag}, delta {r.delta:+.3f} "
                 f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]): {sentence}")
    L.append("")

    L += ["## Off-label output rate", ""]
    pivo = summary.pivot_table(index=["task", "model"], columns="arm", values="off_label_rate")
    L += [pivo.round(3).to_markdown(), ""]
    L += ["Reported separately on purpose. A rise here is an instruction-following failure, "
          "not a commitment-reading failure, and must not be folded into the headline.", ""]

    if not cc.empty:
        L += ["## Control-arm agreement with the published logs", ""]
        L += [cc.to_markdown(index=False), ""]
        L += ["This shows comparability with the published run. It is not a reproduction of "
              "the paper's numbers: closed endpoints may have been updated, and the "
              "open-weight models ran locally under transformers for the paper but are served "
              "through OpenRouter here.", ""]

    L += ["## Boundaries (all three are mandatory)", ""]
    L += [f"{i+1}. {b}" for i, b in enumerate(BOUNDARIES)]
    L += ["", "Strongest supportable claim:", "",
          "> On the same instances, translating the comment into Spanish, Romanian and Swahili "
          "changes G1 balanced accuracy by the deltas above under an identical prompt and "
          "pipeline. Labels are unaffected by translation because they come from market "
          "records. This measures robustness to the surface language of machine-translated "
          "text, not transfer to authentic non-English prediction markets.", ""]

    text = "\n".join(L)
    low = text.lower()
    for bad in FORBIDDEN:
        if bad in low:
            raise SystemExit(f"forbidden claim present in summary: {bad!r}")

    p = os.path.join(OUT, f"e9_summary{t}.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {p}")

    # ---- qualitative examples
    files = glob.glob(os.path.join(PRED, "e9d_predictions_*.csv"))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    src = pd.read_csv(os.path.join(OUT, "e9b_source_instances.csv"))
    tr_files = sorted(glob.glob(os.path.join(OUT, "e9b_translations*.csv")))
    tr = pd.concat([pd.read_csv(f) for f in tr_files], ignore_index=True).drop_duplicates(
        ["comment_id", "task", "lang"])

    d["ok"] = d["pred"].astype(str).str.upper() == d["gt"].astype(str).str.upper()
    wide = d.pivot_table(index=["task", "comment_id"], columns="arm", values="ok",
                         aggfunc="mean")
    lang_cols = [c for c in wide.columns if c != "en" and not c.endswith("_rt")]
    if "en" in wide.columns and lang_cols:
        wide["drop"] = wide["en"] - wide[lang_cols].mean(axis=1)
        cand = wide.sort_values("drop", ascending=False).head(8).reset_index()

        E = ["# E9h -- qualitative examples", "",
             "Selected by the largest drop between the English arm and the mean of the "
             "language arms. Values are the fraction of models correct on that arm.", ""]
        for r in cand.itertuples():
            s = src[(src.task == r.task) & (src.comment_id == r.comment_id)]
            if s.empty:
                continue
            s = s.iloc[0]
            E += [f"## {r.task} / `{r.comment_id}` ({s.platform}/{s.topic})", "",
                  f"- **Market question**: {str(s.question)[:300]}",
                  f"- **Gold**: `{s['gt']}`",
                  f"- **English**: {s.text_en}"]
            for lang in sorted(tr[tr.comment_id == r.comment_id].lang.unique()):
                row = tr[(tr.comment_id == r.comment_id) & (tr.task == r.task) &
                         (tr.lang == lang)]
                if row.empty:
                    continue
                row = row.iloc[0]
                E += [f"- **{lang}**: {row.text_translated}",
                      f"- **{lang} back-translated**: {row.text_backtranslated}"]
            frac = {c: (None if pd.isna(getattr(r, c, np.nan)) else round(getattr(r, c), 2))
                    for c in wide.columns if c != "drop"}
            E += ["", f"- Fraction of models correct by arm: `{frac}`", ""]
        pe = os.path.join(OUT, f"e9h_examples{t}.md")
        with open(pe, "w", encoding="utf-8") as f:
            f.write("\n".join(E))
        print(f"wrote {pe}")


if __name__ == "__main__":
    main()
