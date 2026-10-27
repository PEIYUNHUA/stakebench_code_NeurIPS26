"""
E5 (P2-5, zero new inference): is the G3 failure an artifact of the prompting /
task setup rather than a model limitation?

Reviewer concern (TPc3 W2, Meta c.3): "It is not entirely obvious to me that the
LLMs' structural failures on G3 and G4 represent a fundamental limitation in
language understanding rather than an artifact of the simplistic prompting setup."

Instead of (only) running new prompt variants, we exploit protocol variants that
ALREADY exist in the released prediction logs. Three independent knobs:

  (A) In-context exemplars   : G1 task0_0shot  vs  task0_1shot
  (B) Label-space difficulty : G3 4-way task2  vs  binary task2_flip_hold /
                               task2_decrease_hold  (the easiest possible framing)
  (C) Outcome window         : G3 at h=6/12/24/48/72h vs the default

If accuracy/collapse barely move when we (a) add an exemplar, (b) collapse the
label space to two classes, and (c) sweep the outcome horizon over an order of
magnitude, then the failure is not a prompt-formatting artifact.

Inputs : commitbench_v18/commitbench_v18/<model>/<platform>/<topic>/qwen3_*_task*.csv
Outputs: rebuttal/results/e5_{shot,labelspace,horizon}.csv, e5_summary.md
"""
import os, glob, sys, math
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = "commitbench_v18/commitbench_v18"
OUT  = "rebuttal/results"
os.makedirs(OUT, exist_ok=True)
CLASSES4 = ["hold", "flip", "increase", "decrease"]


def binom_p_two_sided(k, n, p=0.5):
    """Two-sided binomial p-value; normal approximation once n is large enough for it."""
    if n == 0 or not (0 < p < 1):
        return float("nan")
    if n > 200:
        z = (k - n * p) / math.sqrt(n * p * (1 - p))
        return math.erfc(abs(z) / math.sqrt(2))
    from math import comb
    probs = [comb(n, i) * p**i * (1 - p)**(n - i) for i in range(n + 1)]
    return min(1.0, sum(q for q in probs if q <= probs[k] * (1 + 1e-9)))


def read(f):
    try:
        return pd.read_csv(f)
    except Exception:
        try:
            return pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


def collect(pattern):
    """Return concat of every (model, platform, topic) csv matching the suffix pattern."""
    rows = []
    for f in glob.glob(f"{ROOT}/*/*/*/{pattern}"):
        parts = f.replace("\\", "/").split("/")
        model, platform, topic = parts[-4], parts[-3], parts[-2]
        d = read(f)
        if d.empty or not {"gt", "pred"}.issubset(d.columns):
            continue
        d["model"], d["platform"], d["topic"] = model, platform, topic
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def g3_metrics(d):
    gt, pred = d["gt"].astype(str), d["pred"].astype(str)
    acc = (gt == pred).mean()
    rec = []
    for c in sorted(set(gt)):
        sub = pred[gt == c]
        if len(sub):
            rec.append((sub == c).mean())
    return dict(n=len(d), acc=acc,
                macro_recall=float(np.mean(rec)) if rec else np.nan,
                collapse=pred.value_counts(normalize=True).max(),
                n_pred_classes=pred.nunique())


# ---------------------------------------------------------------- (A) 0-shot vs 1-shot
print("[A] G1 zero-shot vs one-shot ...", flush=True)
shot_rows = []
z = collect("*_task0_0shot.csv")
o = collect("*_task0_1shot.csv")
if not z.empty and not o.empty:
    for df, tag in ((z, "0shot"), (o, "1shot")):
        df["gt_b"]   = df["gt"].astype(str).str.lower().isin(["true", "1", "yes"])
        df["pred_b"] = df["pred"].astype(str).str.lower().isin(["true", "1", "yes"])
    for model in sorted(set(z.model) & set(o.model)):
        r = {"model": model}
        for df, tag in ((z, "0shot"), (o, "1shot")):
            s = df[df.model == model]
            if s.empty:
                continue
            tp = ((s.gt_b) & (s.pred_b)).sum(); tn = ((~s.gt_b) & (~s.pred_b)).sum()
            npos = s.gt_b.sum(); nneg = (~s.gt_b).sum()
            r[f"n_{tag}"]        = len(s)
            r[f"acc_{tag}"]      = (s.gt_b == s.pred_b).mean()
            r[f"bal_acc_{tag}"]  = np.mean([tp / npos if npos else np.nan,
                                            tn / nneg if nneg else np.nan])
            r[f"posrate_{tag}"]  = s.pred_b.mean()
        if "acc_0shot" in r and "acc_1shot" in r:
            r["d_acc"]     = r["acc_1shot"] - r["acc_0shot"]
            r["d_bal_acc"] = r["bal_acc_1shot"] - r["bal_acc_0shot"]
            shot_rows.append(r)
shot = pd.DataFrame(shot_rows)
if not shot.empty:
    shot.to_csv(f"{OUT}/e5_shot.csv", index=False)
    print(f"    models compared: {len(shot)}  mean d_acc={shot.d_acc.mean():+.4f}")

# ---------------------------------------------------------------- (B) label-space
print("[B] G3 4-way vs binary pairwise ...", flush=True)
ls_rows = []
full = collect("*_task2.csv")
fh   = collect("*_task2_flip_hold.csv")
dh   = collect("*_task2_decrease_hold.csv")
for name, d in (("G3_4way", full), ("G3_flip_vs_hold", fh), ("G3_decrease_vs_hold", dh)):
    if d.empty:
        continue
    for model, s in d.groupby("model"):
        m = g3_metrics(s)
        k = int((s["gt"].astype(str) == s["pred"].astype(str)).sum())
        chance = 0.5 if name != "G3_4way" else float(
            s["gt"].astype(str).value_counts(normalize=True).max())   # majority-class chance
        m.update(setting=name, model=model, chance=chance,
                 p_vs_chance=binom_p_two_sided(k, len(s), chance))
        ls_rows.append(m)
ls = pd.DataFrame(ls_rows)
if not ls.empty:
    ls.to_csv(f"{OUT}/e5_labelspace.csv", index=False)
    print("   ", ls.groupby("setting")[["acc", "collapse"]].mean().round(3).to_dict())

# ---------------------------------------------------------------- (C) horizon sweep
print("[C] G3 outcome-horizon sweep ...", flush=True)
hz_rows = []
for h in ["h6", "h12", "h24", "h48", "h72"]:
    d = collect(f"*_task2_{h}.csv")
    if d.empty:
        continue
    for model, s in d.groupby("model"):
        m = g3_metrics(s); m.update(horizon=h, model=model)
        hz_rows.append(m)
if not full.empty:
    for model, s in full.groupby("model"):
        m = g3_metrics(s); m.update(horizon="default", model=model)
        hz_rows.append(m)
hz = pd.DataFrame(hz_rows)
if not hz.empty:
    hz.to_csv(f"{OUT}/e5_horizon.csv", index=False)

# ---------------------------------------------------------------- summary
L = ["# E5 - Protocol / prompt-setup robustness of the G3 failure (P2-5, zero new inference)\n",
     "Concern (TPc3 W2, Meta c.3): the G3/G4 failures may be an artifact of a simplistic "
     "prompting setup rather than a model limitation. We test three protocol knobs that are "
     "already present in the released prediction logs.\n"]

if not shot.empty:
    L += ["## (A) In-context exemplar: G1 zero-shot vs one-shot",
          "_Precise scope (do not overstate this in the rebuttal): in `common.prompt_task0` vs "
          "`prompt_task0_zeroshot` the ONLY difference is a worked Example block. Both variants "
          "already carry the same task instruction, the same five structural-signal bullets and "
          "the same pre-extracted feature line. So this measures the marginal value of one "
          "exemplar on top of a rich instruction, not instruction-vs-no-instruction._\n",
          "| Model | n | Acc 0-shot | Acc 1-shot | dAcc | BalAcc 0-shot | BalAcc 1-shot | dBalAcc |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in shot.sort_values("model").iterrows():
        L.append(f"| {r['model']} | {int(r['n_0shot'])} | {r['acc_0shot']:.3f} | {r['acc_1shot']:.3f} | "
                 f"{r['d_acc']:+.3f} | {r['bal_acc_0shot']:.3f} | {r['bal_acc_1shot']:.3f} | {r['d_bal_acc']:+.3f} |")
    L += ["",
          f"- Mean effect of adding one in-context exemplar: **dAcc {shot.d_acc.mean():+.4f}**, "
          f"**dBalAcc {shot.d_bal_acc.mean():+.4f}** (n={len(shot)} models).",
          f"- Models improved by >1 point: **{int((shot.d_acc > 0.01).sum())}/{len(shot)}**; "
          f"degraded by >1 point: **{int((shot.d_acc < -0.01).sum())}/{len(shot)}**.", ""]

if not ls.empty:
    L += ["## (B) Label-space difficulty: 4-way G3 vs the easiest binary framing",
          "_flip-vs-hold and decrease-vs-hold are binary label-space variants. The 0.500 column is a "
          "**uniform-random reference, not empirical chance and not a majority baseline**: the binary "
          "subsets are NOT balanced in the released logs (roughly 665/710 flip/hold for GPT-5.5 "
          "against 385/710 for several open models). Do not write 'balanced two-class'._\n",
          "| Setting | models | mean n | mean Acc | chance | mean collapse (top-label share) | mean #pred classes |",
          "|---|---|---|---|---|---|---|"]
    for s in ["G3_4way", "G3_flip_vs_hold", "G3_decrease_vs_hold"]:
        g = ls[ls.setting == s]
        if g.empty:
            continue
        L.append(f"| {s} | {len(g)} | {g.n.mean():.0f} | {g.acc.mean():.3f} | {g.chance.mean():.3f} | "
                 f"{g.collapse.mean():.3f} | {g.n_pred_classes.mean():.1f} |")
    L.append("")
    for s in ["G3_flip_vs_hold", "G3_decrease_vs_hold"]:
        g = ls[ls.setting == s]
        if g.empty:
            continue
        above = int((g.acc > 0.55).sum()); sig = int((g.p_vs_chance < 0.05).sum())
        L.append(f"- **{s}**: mean Acc {g.acc.mean():.3f} against a uniform-random reference of 0.500; "
                 f"{above}/{len(g)} models above 0.55; {sig}/{len(g)} differ from that reference (p<0.05).")
    L += ["- => Reducing the task to a **binary label space** - the simplest framing, with no "
          "multi-label instruction to misparse - does not recover the missing behavioural signal. "
          "This is sensitivity to TASK FORMULATION, not to prompt wording: it does not address "
          "whether a different G3 prompt would change the result.", ""]

if not hz.empty:
    L += ["## (C) Outcome-horizon sweep (does the 48h window drive the result?)",
          "| Horizon | models | mean n | mean Acc | mean Macro-recall | mean collapse |",
          "|---|---|---|---|---|---|"]
    for h in ["h6", "h12", "h24", "h48", "h72", "default"]:
        g = hz[hz.horizon == h]
        if g.empty:
            continue
        L.append(f"| {h} | {len(g)} | {g.n.mean():.0f} | {g.acc.mean():.3f} | "
                 f"{g.macro_recall.mean():.3f} | {g.collapse.mean():.3f} |")
    sw = hz[hz.horizon != "default"]
    if not sw.empty:
        L += ["",
              f"- Across a 12x sweep of the outcome window (6h -> 72h) mean accuracy moves within "
              f"**{sw.groupby('horizon').acc.mean().max() - sw.groupby('horizon').acc.mean().min():.3f}** "
              f"and mean collapse within "
              f"**{sw.groupby('horizon').collapse.mean().max() - sw.groupby('horizon').collapse.mean().min():.3f}**.",
              "- => The result is stable in the action-definition horizon. This is sensitivity to the "
              "LABEL PROTOCOL, not to prompt richness.", ""]

L += ["## Reading",
      "The three knobs (one exemplar, label-space size, outcome horizon) move the G3 result by less "
      "than the gap to competent performance, so the result is not sensitive to those three protocol "
      "dimensions.",
      "",
      "**What this does NOT establish (2026-07-26 audit).** None of the three knobs is a prompt-wording "
      "manipulation, so TPc3 W2 is only partly addressed:",
      "- (A) is a G1 manipulation and says nothing about whether the G3/G4 prompts are reasonable; the "
      "  0-shot and 1-shot G1 prompts differ ONLY by one worked example and both already carry the full "
      "  instruction, the structural-signal bullets and the pre-extracted feature line.",
      "- (B) changes task formulation; (C) changes label protocol.",
      "- The G3 prompt itself contains \"Do NOT default to hold. If the comment shows any doubt "
      "  (-> decrease) or opposite view (-> flip), choose that.\" (`common.py:640`). That instruction is a "
      "  plausible contributor to the low hold recall and is exactly what these knobs cannot rule out.",
      "- The G4 prompt is not manipulated here at all.",
      "",
      "Do NOT write 'the prompting explanation is ruled out', 'each knob should have rescued the result', "
      "or 'a genuine model limitation rather than a prompting artifact'. Report this as PARTIAL "
      "protocol-sensitivity evidence and state that the direct G3/G4 prompt ablation is outstanding."]

open(f"{OUT}/e5_summary.md", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
print(f"\n[written] e5_shot.csv, e5_labelspace.csv, e5_horizon.csv, e5_summary.md")
