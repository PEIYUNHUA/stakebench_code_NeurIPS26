"""
E3 (P1-3): curate 8-10 qualitative examples of committed vs non-committed language,
showing closed + open model failures (esp. G3 action anticipation and G4 odds lead).

Zero new inference: pivots the existing per-model per-sample predictions.

Inputs:  commitbench_v18/commitbench_v18/<model>/<platform>/<topic>/*_task2.csv (G3)
                                                     .../*_task3_time.csv    (G4)
         data/manifold_data + polymarket_data comments_clean.csv (for position_side)
Outputs: rebuttal/results/e3_g3_candidates.csv, e3_examples.md
"""
import os, glob, sys
import pandas as pd, numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = "commitbench_v18/commitbench_v18"
OUT  = "rebuttal/results"
CLOSED = ["Openrouter-GPT-5.5", "Openrouter-CLAUDE-HAIKU-4.5", "Openrouter-GEMINI-2.5-FLASH"]
OPEN_SHOW = ["Qwen3-32B", "Llama3.2-3B", "finma-30b", "Fino1-14B"]
SHOW = CLOSED + OPEN_SHOW

def clip(s, n=180):
    s = " ".join(str(s).split())
    return s[:n] + ("..." if len(s) > n else "")

# ---------- load all G3 predictions (long) ----------
rows = []
for model in sorted(os.listdir(ROOT)):
    for f in glob.glob(f"{ROOT}/{model}/*/*/*_task2.csv"):
        parts = f.replace("\\", "/").split("/")
        platform, topic = parts[-3], parts[-2]
        try:
            d = pd.read_csv(f, dtype={"market_id": str, "created_at": str})
        except Exception:
            continue
        if not {"gt", "pred", "text"}.issubset(d.columns):
            continue
        d = d[["market_id", "created_at", "question", "text", "gt", "pred", "position_size"]].copy()
        d["model"] = model; d["platform"] = platform; d["topic"] = topic
        rows.append(d)
g = pd.concat(rows, ignore_index=True)
g["key"] = g.platform + "|" + g.topic + "|" + g.market_id + "|" + g.created_at
print(f"G3 long rows: {len(g)}, unique samples: {g.key.nunique()}, models: {g.model.nunique()}")

# pivot: one row per sample, a column per model's prediction
g = g[g["gt"].isin(["hold", "flip", "increase", "decrease"])]          # drop junk labels
base = g.drop_duplicates("key").set_index("key")[["platform","topic","question","text","gt","position_size"]]
wide = g.pivot_table(index="key", columns="model", values="pred", aggfunc="first")
S = base.join(wide)
modelcols = [c for c in wide.columns]
showcols = [m for m in SHOW if m in modelcols]
S["n_models"] = S[modelcols].notna().sum(axis=1)
S["show_cov"] = S[showcols].notna().sum(axis=1)
S["n_correct"] = S.apply(lambda r: sum(str(r[m]) == str(r["gt"]) for m in modelcols if pd.notna(r[m])), axis=1)
S["frac_correct"] = S["n_correct"] / S["n_models"]
S["maj_pred"] = S[modelcols].mode(axis=1)[0]
S.reset_index().to_csv(f"{OUT}/e3_g3_candidates.csv", index=False)

# ---------- select illustrative G3 examples (use ["gt"]; ".gt" is a DataFrame method) ----------
def pick(cond, k=2, sort_asc=True, minlen=25):
    sub = S[cond & (S["text"].str.len() >= minlen) & (S["show_cov"] >= 4)].copy()
    sub = sub.sort_values(["frac_correct", "n_models"], ascending=[sort_asc, False])
    return sub.head(k)

gtc, maj = S["gt"], S["maj_pred"]
examples = []
# (a) non-committed / mild language that models OVER-READ as an action (gt=hold, models predict a shift)
for _, r in pick((gtc == "hold") & (maj != "hold") & (S["n_models"] >= 8), k=2, sort_asc=True).iterrows():
    examples.append(("G3 over-read: gold=hold, models predict a shift (false commitment)", r))
# (b) real conviction that models get the DIRECTION wrong (gt=increase, models predict decrease/flip)
for _, r in pick((gtc == "increase") & (maj.isin(["decrease","flip"])) & (S["n_models"] >= 8), k=2, sort_asc=True).iterrows():
    examples.append(("G3 wrong-direction: gold=increase, models predict decrease/flip", r))
# (c) genuine flip that models MISS (gt=flip, models predict hold/increase)
for _, r in pick((gtc == "flip") & (maj.isin(["hold","increase"])) & (S["n_models"] >= 8), k=2, sort_asc=True).iterrows():
    examples.append(("G3 missed-flip: gold=flip, models predict hold/increase", r))
# (d) correct contrast (any class, high agreement)
for _, r in pick((S["frac_correct"] >= 0.6) & (S["n_models"] >= 8), k=2, sort_asc=False).iterrows():
    examples.append((f"G3 correct contrast (gold={r['gt']})", r))

# ---------- G4 examples ----------
g4 = []
for model in SHOW:
    for f in glob.glob(f"{ROOT}/{model}/*/*/*_task3_time.csv"):
        parts = f.replace("\\", "/").split("/")
        platform, topic = parts[-3], parts[-2]
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if not {"gt_dir", "llm_pred", "stat_2c", "question"}.issubset(d.columns):
            continue
        d = d.copy(); d["model"] = model; d["platform"] = platform; d["topic"] = topic
        g4.append(d)
g4_ex = []
if g4:
    G4 = pd.concat(g4, ignore_index=True)
    G4["k"] = G4.platform + "|" + G4.topic + "|" + G4.question.astype(str) + "|" + G4.window_id.astype(str)
    piv = G4.pivot_table(index="k", columns="model", values="llm_pred", aggfunc="first")
    meta = G4.drop_duplicates("k").set_index("k")[
        ["platform","topic","question","n_comments","avg_odds","next_odds","gt_dir","stat_2c","pos_frac"]]
    G4S = meta.join(piv)
    mcol = [c for c in piv.columns]
    # baseline correct but all LLMs wrong (market-efficiency / LLM-miss illustration)
    G4S["llm_all_wrong"] = G4S.apply(lambda r: all(str(r[m]) != str(r["gt_dir"]) for m in mcol if pd.notna(r[m])), axis=1)
    G4S["base_correct"] = G4S.stat_2c == G4S.gt_dir
    cand = G4S[(G4S.n_comments >= 5) & G4S.base_correct & G4S.llm_all_wrong].head(2)
    for _, r in cand.iterrows(): g4_ex.append(r)
    cand2 = G4S[(G4S.n_comments >= 5) & (~G4S.base_correct)].head(1)
    for _, r in cand2.iterrows(): g4_ex.append(r)

# ---------- render markdown ----------
L = ["# DEPRECATED - E3 candidate-generation output only (P1-3)\n",
     "> **DEPRECATED: candidate-generation output. Do not quote any example from this file without\n"
     "> verifying it against `e3b_verified_examples.md`.** (2026-07-26 audit; banner reinforced\n"
     "> 2026-07-27.)\n"
     "> Two defects make raw candidates unsafe to cite:\n"
     "> 1. **Model coverage.** Only 94 of ~5,185 G3 instances were scored by all 15 models; most\n"
     ">    appear in 3/4/8/12 model logs. The 'x/N' below uses whatever models happen to have the\n"
     ">    instance, so it is not an 'x/15' statement.\n"
     "> 2. **Gold-label inconsistency.** For 107 instances the released logs carry DIFFERENT gold\n"
     ">    labels in the closed-model and open-model runs (Polymarket 1.1%, Manifold 2.5%). An\n"
     ">    affected example has no single gold label and its correct-count cannot be reproduced.\n"
     ">    Two previously quoted examples (Kanye $4,714; Jimmy Carter centenarian) are affected.\n"
     "> `e3b_verified_examples.md` contains the 49 instances that pass both screens.\n"
     "> (Metric impact of the inconsistency is bounded in `e3c_gold_impact.md`: <=0.003 pooled.)\n",
     "Committed vs non-committed language; closed + open model failures. "
     "Auto-selected from existing predictions; verify with E3b, then hand-trim to 8-10.\n"]
L.append("## G3 - Action anticipation (flip / increase / decrease / hold)\n")
for i, (label, r) in enumerate(examples, 1):
    L.append(f"### Example {i} - {label}")
    L.append(f"- **Market** ({r.platform}/{r.topic}): {clip(r.question,120)}")
    L.append(f"- **Comment**: \"{clip(r.text)}\"")
    sz = r.position_size
    L.append(f"- **Commenter stake**: ${sz}" if pd.notna(sz) else "- **Commenter stake**: n/a")
    L.append(f"- **Gold action**: `{r['gt']}` (UNVERIFIED - may differ across model logs, see E3b)"
             f"   |   models correct: {int(r.n_correct)}/{int(r.n_models)} "
             f"(over the {int(r.n_models)} models holding this instance, NOT out of 15)")
    preds = " · ".join(f"{m.replace('Openrouter-','')}=`{r[m]}`" for m in SHOW if m in S.columns and pd.notna(r[m]))
    L.append(f"- **Predictions**: {preds}")
    L.append("")
if g4_ex:
    L.append("## G4 - Collective odds projection (UP / DOWN / NEUTRAL)\n")
    for i, r in enumerate(g4_ex, 1):
        L.append(f"### G4 Example {i} ({r.platform}/{r.topic})")
        L.append(f"- **Market**: {clip(r.question,120)}  ({int(r.n_comments)} window comments, pos_frac={r.pos_frac:.2f})")
        L.append(f"- **Odds move**: {r.avg_odds:.3f} -> {r.next_odds:.3f}  => gold `{r.gt_dir}`")
        L.append(f"- **Stake-weighted baseline**: `{r.stat_2c}` ({'correct' if r.stat_2c==r.gt_dir else 'wrong'})")
        preds = " · ".join(f"{m.replace('Openrouter-','')}=`{r[m]}`" for m in SHOW if m in r.index and pd.notna(r[m]))
        L.append(f"- **LLM predictions**: {preds}")
        L.append("")

open(f"{OUT}/e3_examples.md", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L[:60]))
print(f"\n[written] {OUT}/e3_g3_candidates.csv ({len(S)} samples), e3_examples.md "
      f"({len(examples)} G3 + {len(g4_ex)} G4 examples)")
