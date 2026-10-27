"""
E1b: multiplicity correction and effect distribution for the Wrong-Bettor Test.

Why this exists
---------------
E1 reported "25 of 116 cells significant at p<0.05" plus the five highest follow-rates.
Reporting only uncorrected hits and the top of the distribution invites a
cherry-picking objection. This script adds:
  - Benjamini-Hochberg FDR and Bonferroni corrections over the 116 eligible tests
  - the macro / median / n-weighted mean follow-rate across all cells
  - how many cells sit above 0.5 at all
so the claim can be stated as "localized" rather than "global".

Output: rebuttal/results/e1b_multiplicity.md, e1b_wrongbettor_corrected.csv
"""
import sys
import numpy as np
import pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

OUT = "rebuttal/results"
df = pd.read_csv(f"{OUT}/e1_commitment_validity_rows.csv")
e = df[df.wb_pct_follows_commit.notna() & df.wb_binom_p.notna()].copy()
e["wb_pct_follows_commit"] = e.wb_pct_follows_commit.astype(float)
e["wb_binom_p"] = e.wb_binom_p.astype(float)
m = len(e)

# ---- Benjamini-Hochberg ----
e = e.sort_values("wb_binom_p").reset_index(drop=True)
e["rank"] = np.arange(1, m + 1)
e["bh_crit"] = e["rank"] / m * 0.05
passed = e.wb_binom_p <= e.bh_crit
kmax = int(np.where(passed)[0].max()) + 1 if passed.any() else 0
e["bh_sig"] = False
if kmax:
    e.loc[: kmax - 1, "bh_sig"] = True
# BH q-values (step-up, monotone)
q = (e.wb_binom_p * m / e["rank"]).values
e["bh_q"] = np.minimum.accumulate(q[::-1])[::-1]
e["bonferroni_sig"] = e.wb_binom_p < 0.05 / m

r, w = e.wb_pct_follows_commit, e.wb_n_dir.astype(float)
e.to_csv(f"{OUT}/e1b_wrongbettor_corrected.csv", index=False)

L = []
L.append("# E1b - Wrong-Bettor Test: multiplicity correction and effect distribution\n")
L.append("| Quantity | Value |")
L.append("|---|---|")
L.append(f"| Eligible model-topic tests | {m} |")
L.append(f"| Uncorrected p<0.05 | {int((e.wb_binom_p < .05).sum())} |")
L.append(f"| Benjamini-Hochberg q<0.05 | {int(e.bh_sig.sum())} |")
L.append(f"| Bonferroni p<0.05/{m} | {int(e.bonferroni_sig.sum())} |")
L.append(f"| Macro mean follow-rate | {r.mean():.3f} |")
L.append(f"| Median follow-rate | {r.median():.3f} |")
L.append(f"| n-weighted mean follow-rate | {np.average(r, weights=w):.3f} |")
L.append(f"| Cells above 0.5 | {int((r > .5).sum())} / {m} |")
L.append("")
L.append("## Cells surviving BH-FDR (q<0.05)\n")
L.append("| Model | Platform | Topic | Follow-rate | n | raw p | BH q |")
L.append("|---|---|---|---|---|---|---|")
for _, x in e[e.bh_sig].sort_values("wb_pct_follows_commit", ascending=False).iterrows():
    L.append(f"| {x.model} | {x.platform} | {x.topic} | {x.wb_pct_follows_commit:.3f} | "
             f"{int(x.wb_n_dir)} | {x.wb_binom_p:.4f} | {x.bh_q:.4f} |")
L.append("")
by_split = e[e.bh_sig].groupby(["platform", "topic"]).size()
L.append("Concentration of surviving cells: " +
         ", ".join(f"{p}/{t} = {n}" for (p, t), n in by_split.items()) + "\n")
L.append("## Highest raw follow-rates, with correction status\n")
L.append("| Model | Platform / topic | Follow-rate | n | raw p | BH q | survives BH |")
L.append("|---|---|---|---|---|---|---|")
for _, x in e.sort_values("wb_pct_follows_commit", ascending=False).head(8).iterrows():
    L.append(f"| {x.model} | {x.platform}/{x.topic} | {x.wb_pct_follows_commit:.3f} | {int(x.wb_n_dir)} | "
             f"{x.wb_binom_p:.4f} | {x.bh_q:.4f} | {'yes' if x.bh_sig else 'no'} |")
L.append("")
L.append("**Reading.** The aggregate follow-rate is at chance (macro mean "
         f"{r.mean():.3f}, {int((r > .5).sum())}/{m} cells above 0.5). The commitment signal that "
         "survives multiplicity correction is concentrated in a small number of "
         "model-topic cells, so it should be described as localized and directly "
         "detectable where present, not as a uniform property of the discourse.")
open(f"{OUT}/e1b_multiplicity.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
