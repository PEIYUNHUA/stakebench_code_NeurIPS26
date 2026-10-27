"""
E2b: corrected activity-skew stress test -- comment-retention vs user-retention.

Why this exists
---------------
`e2_stratification.py` takes the (1 - 0.141) quantile of `n_bets` over G3 COMMENT
INSTANCE rows.  That retains ~14.1% of comment instances, but only ~2.9% of unique
commenters, because the most active users contribute many comments each.  The
rebuttal previously described this as "keep only the top 14.1% of users", which is
not what the code does.

This script reports BOTH estimands side by side:
  (A) comment-retention  : threshold on the instance-level n_bets distribution
                           -> retains ~14.1% of G3 comment instances   (= original E2)
  (B) user-retention     : rank UNIQUE commenters by n_bets, keep the top 14.1% of
                           users, then take all of their G3 comments

AUDIT NOTE (2026-07-27, issues_new.md P0)
-----------------------------------------
The previous version of this script reported a 15-model MEAN in the table but a
POOLED (all models x instances concatenated) point estimate and CI in the bootstrap
block.  Those are different estimands, and they disagreed in SIGN on collapse
(table: +0.015 / +0.025; pooled: -0.005 / -0.005).  The cluster bootstrap also
re-grouped duplicated user clusters back together via `groupby("user_id")`, which
shrinks the effective number of clusters in every draw.

This version fixes both:
  * ONE estimand throughout -- the mean over models of the per-model difference.
    The bootstrap point estimate is the estimator evaluated on the observed sample,
    so it reproduces the table differences exactly (up to rounding) by construction.
  * Hierarchical user-clustered bootstrap: users are resampled once per draw and the
    SAME user weights are applied to every model; a user drawn m times contributes m
    DISTINCT clusters, so the "top 14.1% of users" rule sees the resample as a sample
    of the same number of clusters.
  * Restriction rules are recomputed inside every draw (they are part of the
    estimator), per model, exactly as in the point estimate.

Each model is measured on the G3 instances present in its own released logs; model
coverage of the released instance pool is not identical, so the table is a mean over
models, not a comparison on one fixed instance set.

Output: rebuttal/results/e2b_user_retention.csv, e2b_summary.md
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
POLY_RETENTION = 0.141
NBOOT = int(os.environ.get("E2B_NBOOT", "2000"))
RNG = np.random.default_rng(20260727)


def load_model_g3(model, platform="manifold"):
    frames = []
    for f in glob.glob(f"{ROOT}/{model}/{platform}/*/*_task2.csv"):
        try:
            d = pd.read_csv(f, dtype={"market_id": str, "created_at": str})
        except Exception:
            continue
        if {"gt", "pred"}.issubset(d.columns):
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------- data
print("[1] user-frequency table ...")
pos = pd.read_csv("data/manifold_data/positions_clean.csv",
                  usecols=["user_id", "bet_id"], dtype={"user_id": str, "bet_id": str})
freq = pos.groupby("user_id").bet_id.size().rename("n_bets").reset_index()

com = pd.read_csv("data/manifold_data/comments_clean.csv",
                  usecols=["market_id", "created_at", "user_address"], dtype=str)
com = com.rename(columns={"user_address": "user_id"}).drop_duplicates(["market_id", "created_at"])

models = sorted([d for d in os.listdir(ROOT)
                 if os.path.isdir(f"{ROOT}/{d}") and glob.glob(f"{ROOT}/{d}/manifold/*/*_task2.csv")])
print(f"[2] models: {len(models)}")


# ---------------------------------------------------------------- model tensors
class M:
    """Pre-sorted numeric view of one model's G3 rows (sorted by n_bets ascending).

    Everything the estimator needs is an integer-weighted reduction over these
    arrays, so a bootstrap draw is just a new weight vector -- no row copying.
    """
    __slots__ = ("name", "uidx", "nbets", "gtc", "predc", "corr", "npred",
                 "n_rows", "n_users", "user_first_row")


def build(model, gf, user_index):
    gf = gf.sort_values("n_bets", kind="mergesort").reset_index(drop=True)
    m = M()
    m.name = model
    m.uidx = gf.user_id.map(user_index).to_numpy(dtype=np.int64)
    m.nbets = gf.n_bets.to_numpy(dtype=float)
    # raw string comparison, identical to the metrics() helper in e2_stratification.py
    gt = gf["gt"].astype(str)                 # bracket access: ".gt" is a DataFrame method
    pred = gf["pred"].astype(str)
    m.gtc = gt.map({c: i for i, c in enumerate(CLASSES)}).fillna(4).to_numpy(dtype=np.int64)
    codes, uniq = pd.factorize(pred)
    m.predc, m.npred = codes.astype(np.int64), len(uniq)
    m.corr = (gt.to_numpy() == pred.to_numpy())
    m.n_rows = len(gf)
    m.n_users = gf.user_id.nunique()
    return m


print("[3] loading G3 logs ...")
tensors, per_model_meta, user_index, users_seen = [], [], {}, []
raw = {}
for model in models:
    g = load_model_g3(model)
    if g.empty:
        continue
    g = g.merge(com, on=["market_id", "created_at"], how="left").merge(freq, on="user_id", how="left")
    gf = g[g.n_bets.notna() & g.user_id.notna()].copy()
    if gf.empty:
        continue
    for u in gf.user_id.unique():
        if u not in user_index:
            user_index[u] = len(user_index)
            users_seen.append(u)
    raw[model] = gf

U = len(user_index)
for model, gf in raw.items():
    tensors.append(build(model, gf, user_index))
print(f"    {len(tensors)} models, {U} unique commenters in the union")

# user-level n_bets, aligned to the canonical user index (needed for the top-k rule)
user_nbets = np.zeros(U, dtype=float)
for gf in raw.values():
    uu = gf.groupby("user_id").n_bets.first()
    for u, v in uu.items():
        user_nbets[user_index[u]] = v
# deterministic ranking: highest bet count first, user index as tie-break
rank_order = np.lexsort((np.arange(U), -user_nbets))


# ---------------------------------------------------------------- estimator
def wquantile(v_sorted, w, q):
    """Weighted quantile identical to np.quantile on the weight-expanded array.

    `v_sorted` must be ascending; `w` are non-negative integer weights.
    """
    S = w.sum()
    if S <= 0:
        return np.nan
    cum = np.cumsum(w)
    t = q * (S - 1)
    lo = int(np.floor(t))
    frac = t - lo
    i_lo = int(np.searchsorted(cum, lo, side="right"))
    if frac == 0:
        return v_sorted[i_lo]
    i_hi = int(np.searchsorted(cum, lo + 1, side="right"))
    i_hi = min(i_hi, len(v_sorted) - 1)
    return v_sorted[i_lo] + frac * (v_sorted[i_hi] - v_sorted[i_lo])


def wmetrics(m, w):
    S = w.sum()
    if S <= 0:
        return (np.nan, np.nan, np.nan, 0.0)
    acc = float((w * m.corr).sum() / S)
    den = np.bincount(m.gtc, weights=w, minlength=5)
    num = np.bincount(m.gtc, weights=w * m.corr, minlength=5)
    present = den[:4] > 0
    macro = float(np.mean(num[:4][present] / den[:4][present])) if present.any() else np.nan
    collapse = float(np.bincount(m.predc, weights=w, minlength=m.npred).max() / S)
    return (acc, macro, collapse, float(S))


def topk_user_weights(counts):
    """Weight per canonical user under restriction (B), given cluster counts.

    A user drawn `c` times contributes `c` distinct clusters that all carry the same
    bet count, so the top-k rule may admit only part of them at the boundary.
    Returns an array `sel` with 0 <= sel[u] <= counts[u].
    """
    tot = int(counts.sum())
    k = max(1, int(np.ceil(POLY_RETENTION * tot)))
    ordered = counts[rank_order]
    cum = np.cumsum(ordered)
    take = np.clip(k - (cum - ordered), 0, ordered)
    sel = np.zeros_like(counts)
    sel[rank_order] = take
    return sel


def estimate(counts):
    """Mean over models of (restricted metric - full metric), for one weight vector."""
    selB = topk_user_weights(counts)
    accs, out = {}, {k: [] for k in ("d_cmt_acc", "d_usr_acc", "d_cmt_macro", "d_usr_macro",
                                     "d_cmt_collapse", "d_usr_collapse")}
    levels = {t: [] for t in ("full", "cmt", "usr")}
    shares = {"cmt_c": [], "cmt_u": [], "usr_c": [], "usr_u": []}
    for m in tensors:
        w = counts[m.uidx].astype(float)
        if w.sum() <= 0:
            continue
        full = wmetrics(m, w)
        thr = wquantile(m.nbets, w, 1 - POLY_RETENTION)
        w_cmt = np.where(m.nbets >= thr, w, 0.0)
        cmt = wmetrics(m, w_cmt)
        w_usr = selB[m.uidx].astype(float)
        usr = wmetrics(m, w_usr)
        if np.isnan(full[1]) or np.isnan(cmt[1]) or np.isnan(usr[1]):
            continue
        levels["full"].append(full); levels["cmt"].append(cmt); levels["usr"].append(usr)
        out["d_cmt_acc"].append(cmt[0] - full[0]);        out["d_usr_acc"].append(usr[0] - full[0])
        out["d_cmt_macro"].append(cmt[1] - full[1]);      out["d_usr_macro"].append(usr[1] - full[1])
        out["d_cmt_collapse"].append(cmt[2] - full[2]);   out["d_usr_collapse"].append(usr[2] - full[2])
        shares["cmt_c"].append(cmt[3] / full[3]); shares["usr_c"].append(usr[3] / full[3])
        n_u_full = np.unique(m.uidx[w > 0]).size
        shares["cmt_u"].append(np.unique(m.uidx[w_cmt > 0]).size / n_u_full)
        shares["usr_u"].append(np.unique(m.uidx[w_usr > 0]).size / n_u_full)
    accs["diffs"] = {k: float(np.mean(v)) for k, v in out.items() if v}
    accs["levels"] = {t: tuple(float(np.mean([x[i] for x in v])) for i in range(3))
                      for t, v in levels.items() if v}
    accs["shares"] = {k: float(np.mean(v)) for k, v in shares.items() if v}
    accs["per_model"] = out
    return accs


ones = np.ones(U, dtype=np.int64)
POINT = estimate(ones)
LAB = ["d_cmt_macro", "d_usr_macro", "d_cmt_collapse", "d_usr_collapse"]

# ---------------------------------------------------------------- per-model CSV
rows = []
selB0 = topk_user_weights(ones)
for m in tensors:
    w = ones[m.uidx].astype(float)
    full = wmetrics(m, w)
    thr = wquantile(m.nbets, w, 1 - POLY_RETENTION)
    w_cmt = np.where(m.nbets >= thr, w, 0.0)
    cmt = wmetrics(m, w_cmt)
    w_usr = selB0[m.uidx].astype(float)
    usr = wmetrics(m, w_usr)
    rows.append(dict(
        model=m.name, n_full=m.n_rows, n_users_full=m.n_users,
        n_cmt=int(cmt[3]), cmt_share=cmt[3] / full[3],
        cmt_users=int(np.unique(m.uidx[w_cmt > 0]).size),
        cmt_user_share=np.unique(m.uidx[w_cmt > 0]).size / m.n_users,
        n_usr=int(usr[3]), usr_share=usr[3] / full[3],
        usr_users=int(np.unique(m.uidx[w_usr > 0]).size),
        usr_user_share=np.unique(m.uidx[w_usr > 0]).size / m.n_users,
        full_acc=full[0], full_macro=full[1], full_collapse=full[2],
        cmt_acc=cmt[0], cmt_macro=cmt[1], cmt_collapse=cmt[2],
        usr_acc=usr[0], usr_macro=usr[1], usr_collapse=usr[2],
        d_cmt_macro=cmt[1] - full[1], d_usr_macro=usr[1] - full[1],
        d_cmt_collapse=cmt[2] - full[2], d_usr_collapse=usr[2] - full[2],
    ))
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/e2b_user_retention.csv", index=False)

# ---------------------------------------------------------------- bootstrap
print(f"[4] hierarchical user-clustered bootstrap ({NBOOT} draws, {U} clusters) ...")
draws = []
for b in range(NBOOT):
    pick = RNG.integers(0, U, size=U)
    counts = np.bincount(pick, minlength=U)
    d = estimate(counts)["diffs"]
    draws.append([d.get(l, np.nan) for l in LAB])
    if (b + 1) % 200 == 0:
        print(f"    {b + 1}/{NBOOT}", flush=True)
D = np.array(draws, dtype=float)
ci = {l: (float(np.nanpercentile(D[:, i], 2.5)), float(np.nanpercentile(D[:, i], 97.5)))
      for i, l in enumerate(LAB)}

# ---------------------------------------------------------------- summary
lv, dif, sh = POINT["levels"], POINT["diffs"], POINT["shares"]
L = []
L.append("# E2b - activity-skew stress test: comment-retention vs user-retention\n")
L.append("Corrects the description of the original E2 restriction. The original threshold is a")
L.append("quantile over G3 **comment instances**, which retains ~14.1% of instances but only a few")
L.append("percent of unique commenters. Both estimands are reported below.\n")
L.append("**Estimand (2026-07-27 correction).** Every number in this file -- table entries, point")
L.append("estimates and confidence intervals -- is the **mean over the %d models of that model's own"
         % len(R))
L.append("quantity**. Differences are computed per model first and then averaged, so the bootstrap")
L.append("point estimate equals the difference of the table rows by construction. The bootstrap")
L.append("resamples commenters with replacement (a user drawn m times contributes m distinct")
L.append("clusters), applies the same user weights to every model, and recomputes both restriction")
L.append("rules inside each draw. Each model is scored on the G3 instances in its own released logs.\n")
K_SEL = int(topk_user_weights(ones).sum())
L.append("## The two restrictions\n")
L.append("Restriction (B) is now defined **once on the canonical commenter population** -- the")
L.append("%d unique Manifold commenters appearing in any model's G3 log -- and the same %d users"
         % (U, K_SEL))
L.append("(%.1f%%, ranked by lifetime bet count) are used for every model. Restriction (A) keeps the"
         % (100.0 * K_SEL / U))
L.append("original per-model comment-instance quantile, so that the number the earlier reply quoted")
L.append("stays reproducible.\n")
L.append("| Design | comments kept | unique users kept |")
L.append("|---|---|---|")
L.append("| (A) comment-retention (original E2) | %.1f%% | %.1f%% |" % (100*sh["cmt_c"], 100*sh["cmt_u"]))
L.append("| (B) user-retention (top %.1f%% of the canonical commenter population) | %.1f%% | %.1f%% |"
         % (100.0 * K_SEL / U, 100*sh["usr_c"], 100*sh["usr_u"]))
L.append("")
L.append("Shares are means over models of that model's own log. Under (B) they exceed 14.1% because")
L.append("the most active commenters comment more often and so appear in more of the per-model")
L.append("instance samples: the rule keeps %.1f%% of the commenter population but %.1f%% of an average"
         % (100.0 * K_SEL / U, 100*sh["usr_u"]))
L.append("model's commenters and %.1f%% of its comment instances.\n" % (100*sh["usr_c"]))
L.append("## G3 under each restriction (mean over %d models)\n" % len(R))
L.append("| Set | Acc | Macro-recall | Collapse |")
L.append("|---|---|---|---|")
L.append("| Full Manifold | %.3f | %.3f | %.3f |" % lv["full"])
L.append("| (A) comment-retention ~14.1%% of instances | %.3f | %.3f | %.3f |" % lv["cmt"])
L.append("| (B) user-retention top %.1f%% of the commenter population | %.3f | %.3f | %.3f |"
         % ((100.0 * K_SEL / U,) + lv["usr"]))
L.append("")
L.append("## Differences vs full Manifold, hierarchical user-clustered bootstrap 95%% CI (%d draws)\n" % NBOOT)
L.append("| Quantity | point (mean over models) | table check | 95% CI |")
L.append("|---|---|---|---|")
tab = {"d_cmt_macro": lv["cmt"][1] - lv["full"][1], "d_usr_macro": lv["usr"][1] - lv["full"][1],
       "d_cmt_collapse": lv["cmt"][2] - lv["full"][2], "d_usr_collapse": lv["usr"][2] - lv["full"][2]}
for l in LAB:
    L.append("| %s | %+.3f | %+.3f | [%+.3f, %+.3f] |" % (l, dif[l], tab[l], ci[l][0], ci[l][1]))
L.append("")
L.append("Point estimate and table difference agree to %.1e, as they must under a single estimand.\n"
         % max(abs(dif[l] - tab[l]) for l in LAB))
L.append("## Per-model spread of the difference\n")
L.append("| Quantity | mean | min | max | n models with abs(diff) > 0.05 |")
L.append("|---|---|---|---|---|")
for c in LAB:
    L.append("| %s | %+.3f | %+.3f | %+.3f | %d/%d |" % (c, R[c].mean(), R[c].min(), R[c].max(),
                                                        (R[c].abs() > 0.05).sum(), len(R)))
L.append("")
L.append("## Sign of each interval\n")
for l in LAB:
    verdict = ("contains zero" if ci[l][0] <= 0 <= ci[l][1]
               else ("entirely above zero" if ci[l][0] > 0 else "entirely below zero"))
    L.append("- `%s`: %+.3f [%+.3f, %+.3f] -- %s" % (l, dif[l], ci[l][0], ci[l][1], verdict))
open(f"{OUT}/e2b_summary.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
print("\n-> wrote %s/e2b_summary.md" % OUT)
