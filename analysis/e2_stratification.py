"""
E2 (P0-2): 85.9% data-loss / survivorship-bias sensitivity analysis.

Reviewer concern (Fcai W2, TPc3 W1, Meta d.2): Polymarket API mismatch drops 85.9%
of G3 samples, keeping only 14.1% -> possibly high-frequency/professional traders ->
findings may not generalize; failures may be a data artifact.

Strategy (zero new inference): Manifold retains 86% coverage, so we can OBSERVE the
full trader-frequency distribution there. We:
  (1) stratify Manifold G3 by commenter trading frequency (and position size),
  (2) apply a high-activity threshold retaining ~14.1% of G3 comment instances and
      recompute G3,
  (3) check whether the headline failures (low macro-recall, action-class collapse)
      persist under that activity skew.

AUDIT NOTE (2026-07-26): step (2) is a quantile over comment-instance rows, so it keeps
~14.1% of COMMENTS but only ~2.9% of UNIQUE USERS. It is not "the top 14.1% of users",
and it does not replicate or bound Polymarket's missingness mechanism. See
e2b_user_retention.py for the user-retention estimand and clustered bootstrap CIs.

AUDIT NOTE (2026-07-27): step (1) had the same instance-vs-user problem. `pd.qcut` on
G3 comment rows puts an equal number of COMMENTS in each quartile, not an equal number
of commenters, so the "lowest-activity quartile" was a comment-instance quartile ordered
by the commenter's bet count. Both are now produced: the original comment-instance
quartiles (`e2_g3_by_frequency_quartile.csv`) and true commenter quartiles cut on the
canonical unique-commenter population (`e2_g3_by_user_quartile.csv`). Only the latter
supports a statement about lower-activity *participants*.

Inputs:
  data/manifold_data/positions_clean.csv     (user_id, bet_id, contract_id, amount, position_size)
  data/manifold_data/comments_clean.csv       (market_id, created_at, text, user_address)
  commitbench_v18/commitbench_v18/<model>/manifold/<topic>/*_task2.csv  (gt, pred, has_position, position_size)

Outputs (rebuttal/results/):
  e2_user_frequency.csv, e2_g3_by_frequency_quartile.csv,
  e2_highfreq_vs_full.csv, e2_g3_by_positionsize.csv, e2_summary.md
"""
import os, glob, sys
import numpy as np
import pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = "commitbench_v18/commitbench_v18"
OUT  = "rebuttal/results"
os.makedirs(OUT, exist_ok=True)
CLASSES = ["hold", "flip", "increase", "decrease"]
POLY_RETENTION = 0.141   # Polymarket keeps 14.1% of positioned comments

# ---------- 1. per-user trading frequency (Manifold) ----------
print("[1] building user-frequency table from positions_clean.csv ...")
pos = pd.read_csv("data/manifold_data/positions_clean.csv",
                  usecols=["user_id", "bet_id", "contract_id", "amount", "position_size"],
                  dtype={"user_id": str, "bet_id": str, "contract_id": str})
pos["amount"] = pd.to_numeric(pos["amount"], errors="coerce")
freq = pos.groupby("user_id").agg(
    n_bets=("bet_id", "size"),
    n_markets=("contract_id", "nunique"),
    total_amount=("amount", "sum"),
).reset_index()
freq.to_csv(f"{OUT}/e2_user_frequency.csv", index=False)
print(f"    users with bet history: {len(freq)},  median n_bets={freq.n_bets.median():.0f}, "
      f"p90={freq.n_bets.quantile(.9):.0f}, max={freq.n_bets.max():.0f}")

# ---------- 2. comment -> user map ----------
com = pd.read_csv("data/manifold_data/comments_clean.csv",
                  usecols=["market_id", "created_at", "user_address"], dtype=str)
com = com.rename(columns={"user_address": "user_id"})
freq_map = freq.set_index("user_id")

# ---------- metric helpers ----------
def metrics(df):
    n = len(df)
    if n == 0:
        return dict(n=0, acc=np.nan, macro_recall=np.nan, collapse=np.nan,
                    **{f"recall_{c}": np.nan for c in CLASSES})
    gt, pred = df["gt"], df["pred"]           # bracket access: ".gt" is a DataFrame method
    acc = (gt == pred).mean()
    recalls = {}
    for c in CLASSES:
        sub = df[gt == c]
        recalls[f"recall_{c}"] = np.nan if len(sub) == 0 else (sub["pred"] == c).mean()
    present = [recalls[f"recall_{c}"] for c in CLASSES if not np.isnan(recalls[f"recall_{c}"])]
    macro = float(np.mean(present)) if present else np.nan
    collapse = pred.value_counts(normalize=True).max()   # share of the single most-predicted label
    return dict(n=n, acc=acc, macro_recall=macro, collapse=collapse, **recalls)

def load_model_g3(model, platform):
    files = glob.glob(f"{ROOT}/{model}/{platform}/*/*_task2.csv")
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f, dtype={"market_id": str, "created_at": str})
        except Exception:
            continue
        if {"gt", "pred"}.issubset(d.columns):
            d["topic"] = os.path.basename(os.path.dirname(f))
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

models = sorted([d for d in os.listdir(ROOT)
                 if os.path.isdir(f"{ROOT}/{d}") and glob.glob(f"{ROOT}/{d}/manifold/*/*_task2.csv")])
print(f"[2] models with Manifold G3: {len(models)}")

# ---------- 3. frequency-quartile stratification (Manifold, pooled over models) ----------
print("[3] frequency-stratified G3 ...")
q_rows, hf_rows, size_rows, uq_rows = [], [], [], []
pooled_hf, pooled_full = [], []
per_model_freq = {}

for model in models:
    g = load_model_g3(model, "manifold")
    if g.empty:
        continue
    g = g.merge(com, on=["market_id", "created_at"], how="left")
    g = g.merge(freq, on="user_id", how="left")
    cov = g.user_id.notna().mean()
    g_freq = g[g.n_bets.notna()].copy()
    if len(g_freq) == 0:
        continue
    per_model_freq[model] = g_freq
    # (i) COMMENT-INSTANCE quartiles ordered by the commenter's bet count.
    #     Equal number of comments per quartile, NOT equal number of commenters.
    try:
        g_freq["freq_q"] = pd.qcut(g_freq.n_bets, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
    except ValueError:
        g_freq["freq_q"] = "all"
    for q, sub in g_freq.groupby("freq_q", observed=True):
        m = metrics(sub); m.update(model=model, stratum=str(q),
                                   median_nbets=float(sub.n_bets.median()),
                                   n_users=int(sub.user_id.nunique()))
        q_rows.append(m)
    # simulate Polymarket: keep only top-14.1% highest-frequency traders
    thr = g_freq.n_bets.quantile(1 - POLY_RETENTION)
    hf = g_freq[g_freq.n_bets >= thr]
    mf = metrics(g_freq); mh = metrics(hf)
    hf_rows.append(dict(model=model, coverage=round(cov, 3),
                        full_n=mf["n"], full_acc=mf["acc"], full_macro=mf["macro_recall"], full_collapse=mf["collapse"],
                        hf_n=mh["n"], hf_acc=mh["acc"], hf_macro=mh["macro_recall"], hf_collapse=mh["collapse"],
                        d_acc=mh["acc"]-mf["acc"], d_macro=mh["macro_recall"]-mf["macro_recall"],
                        d_collapse=mh["collapse"]-mf["collapse"]))
    pooled_full.append(mf); pooled_hf.append(mh)

# ---------- 3b. TRUE commenter quartiles (equal number of users per stratum) ----------
# The quartiles above hold the number of comments constant. To say anything about
# lower-activity *participants*, the cut has to be made on the commenter population.
# It is defined once, on the union of commenters across models, so every model is
# stratified by the same boundaries.
print("[3b] user-level activity quartiles ...")
if per_model_freq:
    canon = (pd.concat([d[["user_id", "n_bets"]] for d in per_model_freq.values()],
                       ignore_index=True)
               .drop_duplicates("user_id").reset_index(drop=True))
    canon["user_q"] = pd.qcut(canon.n_bets.rank(method="first"), 4,
                              labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    umap = canon.set_index("user_id").user_q
    ubets = canon.groupby("user_q", observed=True).n_bets.median()
    ucount = canon.user_q.value_counts()
    print("    canonical commenters: %d; per quartile: %s"
          % (len(canon), dict(ucount.sort_index())))
    for model, d in per_model_freq.items():
        dd = d.assign(user_q=d.user_id.map(umap))
        for q, sub in dd.groupby("user_q", observed=True):
            m = metrics(sub); m.update(model=model, stratum=str(q),
                                       n_users=int(sub.user_id.nunique()),
                                       median_nbets=float(sub.n_bets.median()))
            uq_rows.append(m)

# ---------- 4. position-size stratification (BOTH platforms; no join needed) ----------
print("[4] position-size-stratified G3 (both platforms) ...")
for platform in ["manifold", "polymarket"]:
    for model in sorted(set(models) | {d for d in os.listdir(ROOT)
                                       if glob.glob(f"{ROOT}/{d}/{platform}/*/*_task2.csv")}):
        g = load_model_g3(model, platform)
        if g.empty or "position_size" not in g.columns:
            continue
        g["position_size"] = pd.to_numeric(g["position_size"], errors="coerce")
        gs = g[g.position_size.notna() & (g.position_size > 0)].copy()
        if len(gs) < 20:
            continue
        try:
            gs["size_q"] = pd.qcut(gs.position_size, 4, labels=["Q1_small","Q2","Q3","Q4_large"], duplicates="drop")
        except ValueError:
            continue
        for q, sub in gs.groupby("size_q", observed=True):
            m = metrics(sub); m.update(model=model, platform=platform, stratum=str(q),
                                       median_size=float(sub.position_size.median()))
            size_rows.append(m)

# ---------- write ----------
pd.DataFrame(q_rows).to_csv(f"{OUT}/e2_g3_by_frequency_quartile.csv", index=False)
pd.DataFrame(uq_rows).to_csv(f"{OUT}/e2_g3_by_user_quartile.csv", index=False)
pd.DataFrame(hf_rows).to_csv(f"{OUT}/e2_highfreq_vs_full.csv", index=False)
pd.DataFrame(size_rows).to_csv(f"{OUT}/e2_g3_by_positionsize.csv", index=False)

# ---------- summary ----------
def pooled_mean(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
    return float(np.mean(v)) if v else float("nan")

L = []
L.append("# E2 - 85.9% data-loss / survivorship-bias sensitivity (P0-2)\n")
L.append("> **READ THIS BEFORE QUOTING ANY NUMBER BELOW (2026-07-26 audit).**\n"
         "> The restriction below is a quantile taken over G3 **comment-instance rows**, so it retains\n"
         "> ~14.1% of *comments* but only ~2.9% of *unique users*. It is NOT 'the top 14.1% of users'.\n"
         "> It also does NOT replicate or bound Polymarket's missingness: the unmatched records'\n"
         "> labels are unobservable. Describe this only as an **activity-skew stress test**.\n"
         "> For the user-retention design and clustered bootstrap CIs, see `e2b_summary.md`.\n")
L.append("**Design.** Manifold retains 86% coverage, so its full trader-activity "
         "distribution is observable. We stratify Manifold G3 by commenter trading frequency, "
         "then apply a high-activity threshold retaining ~14.1% of comment instances and re-measure. "
         "If the failures persist, they are not produced by this particular activity skew.\n")

# quartile table (pooled mean across models)
qdf = pd.DataFrame(q_rows)
if not qdf.empty:
    L.append("## G3 by COMMENT-INSTANCE quartile, ordered by the commenter's bet count (mean across models)")
    L.append("_Equal number of comments per stratum, not equal number of commenters. "
             "This is what the original E2 computed; see the next table for participant quartiles._\n")
    L.append("| Stratum | mean n comments | mean n users | median bets | Acc | Macro-recall | Collapse (top-label share) |")
    L.append("|---|---|---|---|---|---|---|")
    for q in ["Q1_low", "Q2", "Q3", "Q4_high"]:
        s = qdf[qdf.stratum == q]
        if s.empty: continue
        L.append(f"| {q} | {s.n.mean():.0f} | {s.n_users.mean():.0f} | {s.median_nbets.median():.0f} | "
                 f"{s.acc.mean():.3f} | {s.macro_recall.mean():.3f} | {s.collapse.mean():.3f} |")
    L.append("")

uqdf = pd.DataFrame(uq_rows)
if not uqdf.empty:
    L.append("## G3 by COMMENTER quartile (equal number of commenters per stratum, mean across models)")
    L.append(f"_Quartiles cut once on the canonical population of {len(canon)} unique Manifold "
             "commenters by lifetime bet count, then mapped back to their comments. This is the "
             "stratification that supports a statement about lower-activity participants._\n")
    L.append("| Stratum | commenters in population | mean n comments | mean n users in log | median bets | Acc | Macro-recall | Collapse |")
    L.append("|---|---|---|---|---|---|---|---|")
    for q in ["Q1_low", "Q2", "Q3", "Q4_high"]:
        s = uqdf[uqdf.stratum == q]
        if s.empty: continue
        L.append(f"| {q} | {int(ucount.get(q, 0))} | {s.n.mean():.0f} | {s.n_users.mean():.0f} | "
                 f"{s.median_nbets.median():.0f} | {s.acc.mean():.3f} | {s.macro_recall.mean():.3f} | "
                 f"{s.collapse.mean():.3f} |")
    L.append("")

# high-freq vs full headline
L.append("## High-activity subset (~14.1% of comment instances) vs full Manifold")
hfdf = pd.DataFrame(hf_rows)
if not hfdf.empty:
    L.append(f"- Comment->user join coverage: mean **{hfdf.coverage.mean()*100:.1f}%**")
    L.append(f"- **Full Manifold** (pooled mean): Acc **{pooled_mean(pooled_full,'acc'):.3f}**, "
             f"Macro-recall **{pooled_mean(pooled_full,'macro_recall'):.3f}**, "
             f"Collapse **{pooled_mean(pooled_full,'collapse'):.3f}**")
    L.append(f"- **Top-14.1% high-frequency** (pooled mean): Acc **{pooled_mean(pooled_hf,'acc'):.3f}**, "
             f"Macro-recall **{pooled_mean(pooled_hf,'macro_recall'):.3f}**, "
             f"Collapse **{pooled_mean(pooled_hf,'collapse'):.3f}**")
    L.append(f"- **Mean shift high-freq minus full:** dAcc **{hfdf.d_acc.mean():+.3f}**, "
             f"dMacro-recall **{hfdf.d_macro.mean():+.3f}**, dCollapse **{hfdf.d_collapse.mean():+.3f}**")
    L.append("")
    L.append("| Model | full Acc | hf Acc | full Macro | hf Macro | full Collapse | hf Collapse |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in hfdf.iterrows():
        L.append(f"| {r['model']} | {r.full_acc:.3f} | {r.hf_acc:.3f} | {r.full_macro:.3f} | "
                 f"{r.hf_macro:.3f} | {r.full_collapse:.3f} | {r.hf_collapse:.3f} |")
    L.append("")

# interpretation
if not hfdf.empty:
    persists = abs(hfdf.d_macro.mean()) < 0.05 and abs(hfdf.d_collapse.mean()) < 0.05
    L.append("## Interpretation")
    L.append(f"- Restricting Manifold to the high-activity subset (~14.1% of comment instances, "
             f"~2.9% of unique users) shifts macro-recall by "
             f"**{hfdf.d_macro.mean():+.3f}** and collapse by **{hfdf.d_collapse.mean():+.3f}** on average.")
    L.append(f"- => The G3 action-class **{'collapse and low macro-recall PERSIST' if persists else 'metrics move'}** "
             f"under this activity skew. This is a stress test on Manifold, NOT an observation of "
             f"Polymarket's missing 85.9% and NOT a bound on it -- the unmatched records' labels are "
             f"unobservable, so their representativeness remains unresolved.")
    L.append("- See `e2b_summary.md` for the user-retention design and user-clustered bootstrap CIs.")
L.append("")
L.append("## Position-size stratification (cross-check, both platforms) -> e2_g3_by_positionsize.csv")
sdf = pd.DataFrame(size_rows)
if not sdf.empty:
    for platform in ["manifold", "polymarket"]:
        ps = sdf[sdf.platform == platform]
        if ps.empty: continue
        L.append(f"### {platform}")
        L.append("| Size stratum | mean n | median size | Acc | Macro-recall | Collapse |")
        L.append("|---|---|---|---|---|---|")
        for q in ["Q1_small", "Q2", "Q3", "Q4_large"]:
            s = ps[ps.stratum == q]
            if s.empty: continue
            L.append(f"| {q} | {s.n.mean():.0f} | {s.median_size.median():.1f} | {s.acc.mean():.3f} | "
                     f"{s.macro_recall.mean():.3f} | {s.collapse.mean():.3f} |")
        L.append("")

open(f"{OUT}/e2_summary.md", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
print(f"\n[written] e2_user_frequency.csv, e2_g3_by_frequency_quartile.csv, "
      f"e2_highfreq_vs_full.csv, e2_g3_by_positionsize.csv, e2_summary.md")
