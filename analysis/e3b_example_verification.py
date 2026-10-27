"""
E3b: verification pass over the qualitative G3 examples before they go into a reply.

Two problems found in the auto-selected E3 examples
---------------------------------------------------
1. Coverage. Only 94 of ~5,185 G3 instances were scored by ALL 15 models. Most
   instances appear in 3, 4, 8 or 12 model logs. Any "x/15 models" sentence is
   therefore only checkable for that small subset.
2. Gold-label consistency. For a minority of instances the released logs carry
   DIFFERENT gold labels for the closed-model run and the open-model run
   (Polymarket 1.1%, Manifold 2.5% of instances). Both examples previously quoted
   in the rebuttal drafts are in that minority, so their "gold = hold, 0/15" and
   "correct by 14 of 15" descriptions do not match the released logs.

This script rebuilds the example pool with both problems screened out and prints
the exact per-model predictions for each surviving candidate.

Output: rebuttal/results/e3b_verified_examples.md, e3b_gold_consistency.csv
"""
import os, glob, re, sys
import pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = "commitbench_v18/commitbench_v18"
OUT = "rebuttal/results"
CLOSED = ["Openrouter-GPT-5.5", "Openrouter-CLAUDE-HAIKU-4.5", "Openrouter-GEMINI-2.5-FLASH"]
FIN = ["Finance-chat-7B", "Fino1-14B", "Fino1-8B", "finma-30b"]
PROFANE = re.compile(r"(fuck|shit|cunt|bitch|retard|nigg|komrade|kommiela|gerontophile)", re.I)


def load_all():
    frames = []
    for model in sorted(os.listdir(ROOT)):
        for platform in ("manifold", "polymarket"):
            for f in glob.glob(f"{ROOT}/{model}/{platform}/*/*_task2.csv"):
                try:
                    d = pd.read_csv(f, usecols=["market_id", "created_at", "question",
                                                "text", "gt", "pred", "position_size"])
                except Exception:
                    continue
                d["model"] = model
                d["platform"] = platform
                d["topic"] = os.path.basename(os.path.dirname(f))
                frames.append(d)
    a = pd.concat(frames, ignore_index=True)
    a["key"] = (a.platform + "|" + a.market_id.astype(str) + "|"
                + a.created_at.astype(str) + "|" + a.text.astype(str).str[:60])
    return a


print("[1] loading all G3 prediction logs ...")
A = load_all()
cov = A.groupby("key").model.nunique()
gold_n = A.groupby("key")["gt"].nunique()

# ---------- consistency audit ----------
aud = pd.DataFrame({"n_models": cov, "n_gold": gold_n}).reset_index()
aud = aud.merge(A[["key", "platform"]].drop_duplicates("key"), on="key")
aud.to_csv(f"{OUT}/e3b_gold_consistency.csv", index=False)

L = ["# E3b - verified qualitative examples\n"]
L.append("## Screening audit\n")
L.append("| Platform | instances | scored by all 15 models | with >1 distinct gold label |")
L.append("|---|---|---|---|")
for p, g in aud.groupby("platform"):
    L.append(f"| {p} | {len(g)} | {int((g.n_models == 15).sum())} | "
             f"{int((g.n_gold > 1).sum())} ({(g.n_gold > 1).mean():.1%}) |")
L.append("")
L.append("Instances whose gold label differs between the closed-model and open-model runs "
         "cannot be described with a single gold label, and instances not scored by all 15 "
         "models cannot carry an 'x/15' count. Both are excluded below.\n")

keys_ok = set(cov[cov == 15].index) & set(gold_n[gold_n == 1].index)
L.append(f"Usable pool: **{len(keys_ok)}** instances scored by all 15 models with one consistent gold label.\n")

# ---------- candidate examples ----------
rows = []
for key, g in A[A.key.isin(keys_ok)].groupby("key"):
    text = str(g["text"].iloc[0])
    pred_by = {m: g[g.model == m]["pred"].iloc[0] for m in g.model}
    rows.append(dict(
        key=key, platform=g["platform"].iloc[0], topic=g["topic"].iloc[0],
        question=str(g["question"].iloc[0]), text=text,
        stake=float(g["position_size"].iloc[0]), gold=g["gt"].iloc[0],
        n_correct=int((g["gt"] == g["pred"]).sum()),
        profane=bool(PROFANE.search(text)),
        counts=g["pred"].value_counts().to_dict(),
        closed={m: pred_by[m] for m in CLOSED if m in pred_by},
        finance={m: pred_by[m] for m in FIN if m in pred_by},
        general={m: p for m, p in sorted(pred_by.items()) if m not in CLOSED and m not in FIN},
    ))
R = pd.DataFrame(rows)


def block(r, title):
    out = [f"### {title}",
           f"- **Market** ({r.platform}/{r.topic}): {r.question}",
           f"- **Comment**: \"{r.text}\"",
           f"- **Commenter stake**: ${r.stake:,.0f}",
           f"- **Gold action**: `{r.gold}` (identical in all 15 released logs) | "
           f"**models correct: {r.n_correct}/15**",
           f"- **Prediction counts**: {r.counts}",
           f"- **Closed**: " + ", ".join(f"{k.replace('Openrouter-','')}=`{v}`" for k, v in r.closed.items()),
           f"- **Finance-tuned open**: " + ", ".join(f"{k}=`{v}`" for k, v in r.finance.items()),
           f"- **General open-weight**: " + ", ".join(f"{k}=`{v}`" for k, v in r.general.items()), ""]
    return out


clean = R[~R.profane]
L.append("## Over-reading failures (gold = hold, no model predicts hold)\n")
for _, r in clean[(clean.gold == "hold") & (clean.n_correct == 0)].sort_values("stake", ascending=False).iterrows():
    L += block(r, "Over-read: analytical comment, trader in fact held")

L.append("## Wrong-direction failures (0/15 correct)\n")
for _, r in clean[(clean.gold != "hold") & (clean.n_correct == 0)].sort_values("stake", ascending=False).head(3).iterrows():
    L += block(r, f"Wrong direction: gold = {r.gold}")

L.append("## Success contrasts (majority of models correct)\n")
for _, r in clean[clean.n_correct >= 8].sort_values("n_correct", ascending=False).head(4).iterrows():
    L += block(r, f"Success: gold = {r.gold}, {r.n_correct}/15 correct")

open(f"{OUT}/e3b_verified_examples.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L[:40]))
print(f"\n-> wrote {OUT}/e3b_verified_examples.md  ({len(clean)} clean candidates)")
