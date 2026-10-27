"""
E4 (P1-4): G4 market-efficiency confound. Does the G4 failure mean "models are bad"
or "the market already prices the info (no residual signal)"?

We train a SUPERVISED text probe (TF-IDF + logistic / GBDT) to predict the window
odds-direction d_w from the window's comment text, on a SEPARATE full-window pool
reconstructed with common.build_all_windows. We compare:
    probe  vs  stake-weighted baseline  vs  majority-class.

WHAT THESE WINDOWS ARE NOT (2026-07-26 audit, confirmed by E4b)
---------------------------------------------------------------
They are NOT the windows the LLMs were scored on. The reconstruction differs from the
G4 evaluation in window pool (all vs sampled), comment text (full vs first 10 comments
truncated to 120 chars), price-change threshold (1e-9 vs 0.005) and the commitment-
consensus requirement. E4b measured the overlap directly: the market sets coincide
100%, but window-level match is 0.0% on Manifold and 45.0% on Polymarket -- these are
two different segmentations, not the same windows under different filters.

So this is a PRELIMINARY SIGNAL-SENSITIVITY ANALYSIS, not a controlled comparison. A
probe that does not beat the baselines is consistent with limited easily recoverable
lexical signal; it cannot separate market efficiency from LLM input and prompt
limitations, and it measures no efficiency quantity (spread, liquidity, absorption
speed, calibration) at all. Never write "the same windows the LLMs saw", "residual
signal is genuinely limited", or "efficiency gradient".

Pre-specified primary probe: TF-IDF + logistic regression. GBDT is secondary. The
headline must not be whichever of the two scored higher on the test set.

Plus: recompute, from the existing task3 CSVs, how many model-split instances where an
LLM beats the naive baseline fall on Manifold vs Polymarket (paper Table 14, "27/32 on
Manifold") -- a descriptive platform contrast, not a measurement of efficiency.

No new LLM calls (the probes themselves are newly fitted sklearn models).
Outputs: rebuttal/results/e4_probe_results.csv, e4_gradient.csv, e4_summary.md
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, "stakebench_code_NeurIPS26")
# point the benchmark code at this machine's data copy before importing config
os.environ.setdefault("COMMITBENCH_MANIFOLD_DIR",   os.path.abspath("data/manifold_data"))
os.environ.setdefault("COMMITBENCH_POLYMARKET_DIR", os.path.abspath("data/polymarket_data"))
import common
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = "commitbench_v18/commitbench_v18"
OUT  = "rebuttal/results"
PLATS = {
    "manifold":  ("data/manifold_data/comments_clean.csv",  "data/manifold_data/prices"),
    "polymarket":("data/polymarket_data/comments_clean.csv","data/polymarket_data/prices"),
}
COLS = ["market_id","question","resolution","created_at","text","position_side","position_size","prob_at_comment"]

def windows_to_frame(wins):
    rows = []
    for w in wins:
        nxt = w.get("next_window_avg_odds")
        if nxt is None:
            continue
        delta = nxt - w["avg_odds"]
        if abs(delta) < 1e-9:
            continue
        target = "UP" if delta > 0 else "DOWN"
        text = " ".join(str(c.get("text","")) for c in w["comments"])
        yes_w, no_w = w["yes_weight"], w["no_weight"]
        if yes_w == no_w:
            stake = "NA"
        else:
            stake = "UP" if yes_w > no_w else "DOWN"   # stake-weighted majority side -> price moves toward it
        rows.append(dict(market_id=str(w["market_id"]), win_start=w["win_start_ts"],
                         target=target, text=text, stake_pred=stake,
                         n_comments=len(w["comments"]), avg_odds=w["avg_odds"]))
    return pd.DataFrame(rows)

def run_probe(df, platform):
    df = df[df.text.str.len() >= 10].sort_values("win_start").reset_index(drop=True)
    n = len(df)
    if n < 60:
        return dict(platform=platform, n=n, note="too few windows")
    cut = int(n * 0.7)
    tr, te = df.iloc[:cut], df.iloc[cut:]        # time-split (train = earlier windows)
    y_tr, y_te = tr.target.values, te.target.values
    vec = TfidfVectorizer(max_features=6000, ngram_range=(1,2), min_df=2, stop_words="english")
    Xtr = vec.fit_transform(tr.text); Xte = vec.transform(te.text)
    out = dict(platform=platform, n=n, n_train=len(tr), n_test=len(te),
               up_rate_test=float((y_te=="UP").mean()))
    # majority-class baseline
    maj = pd.Series(y_tr).value_counts().idxmax()
    out["acc_majority"] = accuracy_score(y_te, [maj]*len(y_te))
    # stake-weighted baseline (only where defined)
    m = te.stake_pred != "NA"
    out["acc_stake"] = accuracy_score(y_te[m.values], te.stake_pred[m].values) if m.any() else np.nan
    out["stake_cov"] = float(m.mean())
    # Probe 1: TF-IDF + logistic
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(Xtr, y_tr); p_lr = lr.predict(Xte)
    out["acc_probe_LR"] = accuracy_score(y_te, p_lr)
    out["f1_probe_LR"]  = f1_score(y_te, p_lr, pos_label="UP", average="binary")
    # Probe 2: TF-IDF -> SVD -> HistGBDT
    try:
        k = min(120, Xtr.shape[1]-1)
        svd = TruncatedSVD(n_components=k, random_state=0)
        Ztr = svd.fit_transform(Xtr); Zte = svd.transform(Xte)
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0)
        gb.fit(Ztr, y_tr); p_gb = gb.predict(Zte)
        out["acc_probe_GBDT"] = accuracy_score(y_te, p_gb)
    except Exception as e:
        out["acc_probe_GBDT"] = np.nan
    # Paired probe-minus-baseline differences with a market-clustered bootstrap CI.
    # LR is the pre-specified primary probe. Windows from one market share language and
    # price dynamics, so the cluster is the market, not the window. The stake comparison
    # is run on the subset where the stake baseline is defined, for BOTH sides, so the
    # two accuracies are measured on identical rows.
    out.update(paired_ci(te, y_te, p_lr))
    return out


NBOOT_E4 = 2000


def paired_ci(te, y_te, p_lr, nboot=NBOOT_E4, seed=20260727):
    y = np.asarray(y_te)
    p = np.asarray(p_lr)
    maj = np.full(len(y), pd.Series(y).value_counts().idxmax())
    stake = te.stake_pred.values
    has_stake = stake != "NA"
    mkt = te.market_id.values
    idx_by_market = [np.where(mkt == m)[0] for m in pd.unique(mkt)]
    rng = np.random.default_rng(seed)
    d_maj, d_stk = [], []
    for _ in range(nboot):
        pick = rng.integers(0, len(idx_by_market), size=len(idx_by_market))
        ii = np.concatenate([idx_by_market[j] for j in pick])
        d_maj.append((p[ii] == y[ii]).mean() - (maj[ii] == y[ii]).mean())
        jj = ii[has_stake[ii]]
        d_stk.append((p[jj] == y[jj]).mean() - (stake[jj] == y[jj]).mean() if len(jj) else np.nan)
    jj0 = np.where(has_stake)[0]
    return dict(
        n_test_markets=len(idx_by_market),
        d_LR_vs_majority=float((p == y).mean() - (maj == y).mean()),
        d_LR_vs_majority_lo=float(np.nanpercentile(d_maj, 2.5)),
        d_LR_vs_majority_hi=float(np.nanpercentile(d_maj, 97.5)),
        acc_probe_LR_on_stake_subset=float((p[jj0] == y[jj0]).mean()) if len(jj0) else np.nan,
        d_LR_vs_stake=float((p[jj0] == y[jj0]).mean() - (stake[jj0] == y[jj0]).mean()) if len(jj0) else np.nan,
        d_LR_vs_stake_lo=float(np.nanpercentile(d_stk, 2.5)),
        d_LR_vs_stake_hi=float(np.nanpercentile(d_stk, 97.5)),
    )

# ---------- reconstruct windows + run probes ----------
probe_rows = []
for platform, (cpath, pdir) in PLATS.items():
    cache = f"{OUT}/e4_windows_{platform}.csv"
    if os.path.exists(cache):                      # reuse windows built by an earlier run
        print(f"[{platform}] reusing cached windows {cache}", flush=True)
        wdf = pd.read_csv(cache, dtype={"market_id": str})
        wdf["text"] = wdf["text"].fillna("").astype(str)
    else:
        # Reproduce common.load_data's normalization without importing it: the released
        # loader reads market_attrs.csv as utf-8, but this machine's polymarket copy is
        # latin-1. Same steps -- resolution mapped from market_attrs, prob_at_comment
        # interpolated from the price series -- so windows match what the LLMs saw.
        print(f"[{platform}] loading comments ...", flush=True)
        enc = "latin-1" if platform == "polymarket" else "utf-8-sig"
        com = pd.read_csv(cpath, usecols=lambda c: c in COLS + ["resolution"],
                          dtype=str, encoding=enc, engine="python", on_bad_lines="skip")
        com.columns = [c.lstrip("﻿") for c in com.columns]
        com["position_side"] = com["position_side"].fillna("").str.strip().str.upper()
        if "resolution" not in com.columns:
            mk = pd.read_csv(os.path.join(os.path.dirname(cpath), "market_attrs.csv"),
                             encoding=enc, engine="python", on_bad_lines="skip",
                             dtype={"market_id": str})
            mk["resolution"] = mk["resolution"].astype(str).str.strip()
            mk = mk[mk["resolution"].str.upper().isin(["YES", "NO"])]
            com["resolution"] = com["market_id"].map(
                mk.set_index("market_id")["resolution"].str.upper().to_dict())
            com = com[com["resolution"].notna()].reset_index(drop=True)
        com["position_size"] = pd.to_numeric(com.get("position_size"), errors="coerce")
        if "prob_at_comment" not in com.columns:
            print("  estimating prob_at_comment from price series ...", flush=True)
            com = common._add_prob_at_comment(com, pdir)
        com["prob_at_comment"] = pd.to_numeric(com["prob_at_comment"], errors="coerce")
        print(f"  {len(com)} comments; building windows ...", flush=True)
        wins = common.build_all_windows(com, prices_dir=pdir if os.path.isdir(pdir) else "")
        wdf = windows_to_frame(wins)
        wdf.to_csv(cache, index=False)
    print(f"  usable windows (with next-odds & move): {len(wdf)}", flush=True)
    res = run_probe(wdf, platform)
    probe_rows.append(res)
    print("  ", {k: (round(v,3) if isinstance(v,float) else v) for k,v in res.items() if k!='note'}, flush=True)

pd.DataFrame(probe_rows).to_csv(f"{OUT}/e4_probe_results.csv", index=False)

# ---------- efficiency gradient from task3 CSVs ----------
# Paper Table 14 def: "Positive" = model-split instances where LLM exceeds the NAIVE
# odds-direction baseline; Splits = model x topic (one variant). 270 total (Man 180/Poly 90).
grad = []
VARIANT = "time"   # one variant per (model, topic) to match the 180/90 split counts
for f in glob.glob(f"{ROOT}/*/*/*/*_task3_{VARIANT}.csv"):
    parts = f.replace("\\","/").split("/"); model,platform,topic = parts[-4],parts[-3],parts[-2]
    try:
        d = pd.read_csv(f)
    except Exception:
        continue
    if not {"llm_correct","naive_correct"}.issubset(d.columns) or len(d)==0:
        continue
    grad.append(dict(model=model, platform=platform, topic=topic, n=len(d),
                     llm_acc=d["llm_correct"].mean(), naive_acc=d["naive_correct"].mean(),
                     stake_acc=d["stat_correct_2c"].mean() if "stat_correct_2c" in d else np.nan))
gdf = pd.DataFrame(grad)
gdf.to_csv(f"{OUT}/e4_gradient.csv", index=False)

# ---------- summary ----------
L = ["# E4 - G4 separate full-window lexical probe (P1-4)\n",
     "Question (TPc3 W3 / Meta d.4): is G4 failure a model limitation or market efficiency?\n",
     "> **READ THIS BEFORE QUOTING ANY NUMBER BELOW (2026-07-26 / 2026-07-27 audits).**\n"
     "> These windows are NOT the ones the LLMs saw. This probe uses the full reconstructed\n"
     "> window pool and the full window text; the LLM G4 evaluation used sampled windows and\n"
     "> inputs truncated to the first 10 comments at 120 chars, under price-eps 0.005 (vs 1e-9\n"
     "> here) and a commitment-consensus requirement. E4b measured the overlap: market sets\n"
     "> coincide 100%, window-level match is 0.0% (Manifold) / 45.0% (Polymarket). So this is a\n"
     "> **preliminary signal-sensitivity analysis, not a controlled comparison** with the LLMs,\n"
     "> and it cannot separate market efficiency from LLM input/prompt limitations.\n"
     "> Never write 'the same windows/comments the LLMs saw'.\n"
     "> Logistic regression is the PRE-SPECIFIED primary probe; do not headline whichever of\n"
     "> LR/GBDT scored higher on the test set.\n",
     "## Supervised text probe vs baselines (chronological split, full-window text)"]
L.append("| Platform | #windows | test UP-rate | Majority | Stake-wtd | Probe-LR | Probe-GBDT |")
L.append("|---|---|---|---|---|---|---|")
for r in probe_rows:
    if r.get("note"):
        L.append(f"| {r['platform']} | {r['n']} | - | - | - | {r['note']} | |"); continue
    L.append(f"| {r['platform']} | {r['n']} | {r['up_rate_test']:.3f} | {r['acc_majority']:.3f} | "
             f"{r.get('acc_stake',float('nan')):.3f} | {r['acc_probe_LR']:.3f} | {r.get('acc_probe_GBDT',float('nan')):.3f} |")
L.append("")
L.append("## Paired probe-minus-baseline differences, market-clustered bootstrap 95% CI")
L.append("_Pre-specified primary probe = logistic regression. Each comparison is computed on rows "
         "where both predictors are defined, and resampled by market (2,000 draws)._\n")
L.append("| Platform | test markets | LR - majority | 95% CI | LR - stake-wtd (common rows) | 95% CI |")
L.append("|---|---|---|---|---|---|")
for r in probe_rows:
    if r.get("note"): continue
    L.append(f"| {r['platform']} | {r.get('n_test_markets','-')} | {r['d_LR_vs_majority']:+.3f} | "
             f"[{r['d_LR_vs_majority_lo']:+.3f}, {r['d_LR_vs_majority_hi']:+.3f}] | "
             f"{r['d_LR_vs_stake']:+.3f} | [{r['d_LR_vs_stake_lo']:+.3f}, {r['d_LR_vs_stake_hi']:+.3f}] |")
L.append("")
L.append("These intervals quantify the probe-versus-baseline gap that earlier versions of this "
         "analysis reported without any uncertainty. They are a sensitivity check on a separate "
         "window pool; they do not license a statement about market efficiency.\n")
# interpretation
for r in probe_rows:
    if r.get("note"): continue
    # Read the pre-specified primary probe against the STRONGER baseline, and judge it by the
    # paired interval above rather than by the legacy 0.03 decision threshold.
    lr = r.get("acc_probe_LR", float("nan"))
    base_name, base = max([("majority", r.get("acc_majority")), ("stake-weighted", r.get("acc_stake"))],
                          key=lambda t: (t[1] if t[1] == t[1] else -1))
    d, lo, hi = ((r["d_LR_vs_majority"], r["d_LR_vs_majority_lo"], r["d_LR_vs_majority_hi"])
                 if base_name == "majority" else
                 (r["d_LR_vs_stake"], r["d_LR_vs_stake_lo"], r["d_LR_vs_stake_hi"]))
    crosses = lo <= 0 <= hi
    verdict = ("interval excludes zero -> the probe separates from this baseline on this window pool"
               if not crosses else
               "interval contains zero -> no separation from the stronger baseline; consistent with "
               "limited easily recoverable LEXICAL signal (not a measurement of market efficiency)")
    L.append(f"- **{r['platform']}**: primary probe (LR) {lr:.3f} vs stronger baseline "
             f"({base_name}) {base:.3f}; paired difference {d:+.3f} [{lo:+.3f}, {hi:+.3f}] => {verdict}")
    gb = r.get("acc_probe_GBDT", float("nan"))
    if gb == gb and gb > lr:
        L.append(f"  - Secondary GBDT probe scores higher ({gb:.3f} vs {lr:.3f}) but was not "
                 f"pre-specified; do not use it as the headline.")
L.append("")

# gradient (reproduce paper Table 14: Positive = LLM exceeds NAIVE baseline)
if not gdf.empty:
    g = gdf.copy(); g["pos"] = g.llm_acc > g.naive_acc
    L.append("## Platform difference (reproduces paper Table 14 'g4-platform-gains')")
    L.append("| Platform | Splits | Positive (LLM > naive) | Share |")
    L.append("|---|---|---|---|")
    tot_s = tot_p = 0
    for plat in ["manifold", "polymarket"]:
        gp = g[g.platform == plat]; s, p = len(gp), int(gp.pos.sum())
        tot_s += s; tot_p += p
        L.append(f"| {plat} | {s} | {p} | {p/s*100:.1f}% |" if s else f"| {plat} | 0 | 0 | - |")
    L.append(f"| **Total** | {tot_s} | {tot_p} | {tot_p/tot_s*100:.1f}% |")
    man = int(g[g.platform=='manifold'].pos.sum())
    L.append("")
    L.append(f"- **{man} of {tot_p}** positive (LLM-beats-naive) instances are on Manifold.")
    L.append(f"- Mean LLM gain over naive baseline: Manifold **{(g[g.platform=='manifold'].llm_acc-g[g.platform=='manifold'].naive_acc).mean():+.3f}**, "
             f"Polymarket **{(g[g.platform=='polymarket'].llm_acc-g[g.platform=='polymarket'].naive_acc).mean():+.3f}**.")
    L.append("- => CONSISTENT WITH an efficiency-based explanation, but this does NOT identify "
             "efficiency as the cause. Platform, topic composition, window availability and sample "
             "size are jointly confounded; these are model-topic splits, not independent replicates; "
             "and no efficiency quantity (spread, liquidity, absorption speed, calibration) is "
             "measured anywhere in this script.")
    L.append("- Do NOT write 'deliberate efficiency gradient', 'residual signal concentrates', or "
             "reframe the two-platform design as having been chosen to span efficiency regimes.")
open(f"{OUT}/e4_summary.md","w",encoding="utf-8").write("\n".join(L))
print("\n".join(L))
print(f"\n[written] e4_probe_results.csv, e4_gradient.csv, e4_windows_*.csv, e4_summary.md")
