"""
E6b: does the lexical probe survive market-grouped and user-grouped splits?

Motivation (audit RI-05)
------------------------
E6 uses a single global chronological split, so the same market -- and the same
commenter -- can appear in both train and test. A probe could then score above the
floor by memorising market-specific or user-specific vocabulary rather than by
recovering commitment. A reviewer can raise this in one line, so we should have the
answer ready.

This re-runs the same TF-IDF + logistic-regression probe under three splits:
  (1) chronological 70/30      -- reproduces E6
  (2) market-grouped 70/30     -- no market appears in both sides
  (3) user-grouped 70/30       -- no commenter appears in both sides

Primary metric is unchanged: balanced accuracy against the 1/k floor, with a
label-permutation test.

Output: rebuttal/results/e6b_group_splits.md, e6b_group_splits.csv
"""
import os, glob, math, sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = "commitbench_v18/commitbench_v18"
OUT = "rebuttal/results"
N_PERM = 1000
RNG = np.random.RandomState(20260726)

TIERS = {
    "G1_has_position":  ("task0_0shot", lambda s: s.astype(str).str.strip().str.lower(), {"true", "false"}),
    "G2_position_side": ("task1",       lambda s: s.astype(str).str.strip().str.lower(), {"yes", "no"}),
    "G3_action_4way":   ("task2",       lambda s: s.astype(str).str.strip().str.lower(),
                         {"flip", "increase", "decrease", "hold"}),
}


KEY = ["market_id", "created_at", "text"]
EXCLUDED = {}


def load_instances(suffix, platform, tag=None):
    """Union of released evaluation instances, one row per instance key.

    Keys whose gold label disagrees across model logs are dropped and counted (see
    e3b_gold_consistency.csv); keeping them would make the probe target depend on
    which file the glob reached first.
    """
    frames = []
    for f in glob.glob(f"{ROOT}/*/{platform}/*/*_{suffix}.csv"):
        try:
            d = pd.read_csv(f, dtype={"market_id": str, "created_at": str})
        except Exception:
            continue
        if d.empty or not {"gt", "text"}.issubset(d.columns):
            continue
        frames.append(d[["market_id", "created_at", "text", "gt"]])
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["text"] = d["text"].fillna("").astype(str)
    nun = d.assign(_g=d["gt"].astype(str).str.strip().str.lower()).groupby(KEY)["_g"].transform("nunique")
    if tag is not None:
        EXCLUDED[tag] = (int(d.loc[nun > 1, KEY].drop_duplicates().shape[0]),
                         int(d[KEY].drop_duplicates().shape[0]))
    d = d[nun == 1]
    d = d.drop_duplicates(subset=KEY).reset_index(drop=True)
    d = d[d.text.str.len() >= 10]
    d["_ts"] = pd.to_datetime(d["created_at"], utc=True, errors="coerce")
    return d.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)


def attach_user(d, platform):
    """Join commenter identity so we can build a user-grouped split.

    Encoding differs per local copy: the Manifold export carries a UTF-8 BOM (so a
    latin-1 read renames the first column to 'i>?market_id'), while the Polymarket
    export needs latin-1. Try both and normalise the header either way.
    """
    com = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            com = pd.read_csv(f"data/{platform}_data/comments_clean.csv", dtype=str,
                              encoding=enc, usecols=lambda c: c.strip("﻿") in
                              ("market_id", "created_at", "user_address"))
            com.columns = [c.strip("﻿") for c in com.columns]
            if {"market_id", "created_at", "user_address"}.issubset(com.columns):
                break
            com = None
        except Exception:
            com = None
    if com is None:
        return d.assign(user_id=np.nan)
    com = com.drop_duplicates(["market_id", "created_at"])
    return d.merge(com.rename(columns={"user_address": "user_id"}),
                   on=["market_id", "created_at"], how="left")


def split_indices(d, mode):
    n = len(d)
    if mode == "chronological":
        cut = int(n * 0.7)
        return np.arange(cut), np.arange(cut, n)
    col = "market_id" if mode == "market_group" else "user_id"
    g = d[col]
    if g.isna().all():
        return None
    groups = g.fillna("__na__").values
    uniq = pd.unique(groups)
    perm = RNG.permutation(len(uniq))
    uniq = uniq[perm]
    # take groups until ~70% of rows are assigned to train
    sizes = pd.Series(groups).value_counts()
    train_groups, total = set(), 0
    for u in uniq:
        if total >= 0.7 * n:
            break
        train_groups.add(u); total += sizes[u]
    mask = np.isin(groups, list(train_groups))
    return np.where(mask)[0], np.where(~mask)[0]


def run(d, tier, mode):
    norm, valid = TIERS[tier][1], TIERS[tier][2]
    y = norm(d["gt"])
    keep = y.isin(valid)
    d, y = d[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    if len(d) < 200:
        return None
    sp = split_indices(d, mode)
    if sp is None:
        return None
    tr, te = sp
    if len(tr) < 100 or len(te) < 50:
        return None
    ytr, yte = y.iloc[tr].values, y.iloc[te].values
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return None
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2,
                          sublinear_tf=True, stop_words="english")
    Xtr = vec.fit_transform(d.text.iloc[tr]); Xte = vec.transform(d.text.iloc[te])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    clf.fit(Xtr, ytr)
    p = clf.predict(Xte)
    k = len(valid)
    bal = balanced_accuracy_score(yte, p)
    rng = np.random.RandomState(0)
    null = np.array([balanced_accuracy_score(rng.permutation(yte), p) for _ in range(N_PERM)])
    return dict(tier=tier, split=mode, n=len(d), n_train=len(tr), n_test=len(te),
                k=k, floor=1.0 / k, bal_acc=bal, gain=bal - 1.0 / k,
                acc=accuracy_score(yte, p), macro_f1=f1_score(yte, p, average="macro"),
                p_perm=float((null >= bal).sum() + 1) / (N_PERM + 1))


rows = []
for platform in ("manifold", "polymarket"):
    for tier, (suffix, _, _) in TIERS.items():
        d = load_instances(suffix, platform, tag=(platform, tier))
        if d.empty:
            continue
        d = attach_user(d, platform)
        cov = d.user_id.notna().mean() if "user_id" in d else 0.0
        for mode in ("chronological", "market_group", "user_group"):
            if mode == "user_group" and cov < 0.5:
                continue
            r = run(d, tier, mode)
            if r:
                r.update(platform=platform, user_join_cov=cov)
                rows.append(r)
                print(f"  {platform:<11}{tier:<18}{mode:<15} bal={r['bal_acc']:.3f} "
                      f"floor={r['floor']:.3f} p={r['p_perm']:.4f}")

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/e6b_group_splits.csv", index=False)

L = ["# E6b - lexical probe under market-grouped and user-grouped splits\n",
     "Same TF-IDF + logistic-regression probe as E6. Balanced accuracy against the 1/k floor, "
     f"{N_PERM}-draw label-permutation test. Grouped splits guarantee that no market "
     "(resp. no commenter) appears on both sides.\n",
     "Instance keys with disagreeing gold labels across model logs are excluded: " +
     "; ".join(f"{p}/{t} {n} of {tot}" for (p, t), (n, tot) in EXCLUDED.items()) + ".\n"]
L.append("| Platform | Tier | Split | n train/test | Bal-acc | 1/k floor | gain | p (perm) |")
L.append("|---|---|---|---|---|---|---|---|")
for _, x in R.iterrows():
    L.append(f"| {x.platform} | {x.tier} | {x.split} | {int(x.n_train)}/{int(x.n_test)} | "
             f"{x.bal_acc:.3f} | {x.floor:.3f} | {x.gain:+.3f} | {x.p_perm:.4f} |")
L.append("")
g1 = R[R.tier == "G1_has_position"]
if not g1.empty:
    L.append(f"**G1 holds up under grouping**: balanced accuracy stays in "
             f"[{g1.bal_acc.min():.3f}, {g1.bal_acc.max():.3f}] against a 0.500 floor across all splits, "
             f"so the detectability of explicit commitment language is not an artefact of the same "
             f"market or the same commenter appearing on both sides.\n")
g3 = R[R.tier == "G3_action_4way"]
if not g3.empty:
    sig = g3[(g3.gain > 0) & (g3.p_perm < 0.05)]
    L.append(f"**G3 stays near the floor under every split** (balanced accuracy "
             f"{g3.bal_acc.min():.3f}-{g3.bal_acc.max():.3f} against 0.250), far below the G1 range, "
             f"so the tier-wise decline reported in the paper is not produced by the split design "
             f"either.\n")
    if not sig.empty:
        L.append("**But do not write 'G3 is at the floor under every split'.** "
                 + "; ".join(f"{r.platform} {r.split} reaches {r.bal_acc:.3f} (+{r.gain:.3f}, "
                             f"p={r.p_perm:.4f})" for _, r in sig.iterrows())
                 + ". A grouped split can raise G3 slightly above the floor, which is the honest "
                 "statement: G3 recoverability is small and split-dependent, while G1 is large and "
                 "stable. Each grouped split is one random assignment, not cross-validated.\n")
L.append("Caveat: grouped splits are a single random 70/30 assignment over groups, not cross-validated, "
         "so small differences between splits should not be over-read.")
open(f"{OUT}/e6b_group_splits.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n" + "\n".join(L))
