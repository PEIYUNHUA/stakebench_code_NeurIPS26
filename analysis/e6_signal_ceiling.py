"""
E6 (P0-1 motivation, no new LLM calls): how much of each task's label is recoverable
from shallow lexical features alone?

Reviewer concern (Fcai W1, Meta c.1/d.1): "It remains unclear whether this failure
stems from LLM reasoning deficits, or simply because the highly noisy text data
lacks sufficient predictive signal."

We answer it with a SUPERVISED LEXICAL RECOVERABILITY PROBE: a plain bag-of-words
model (TF-IDF -> logistic regression) trained on a 70/30 chronological split of the
union of released evaluation instances, for each task tier:

    G1  has-position?           (binary)
    G2  which side?             (YES / NO)
    G3  next action             (flip / increase / decrease / hold)
    G3b flip-vs-hold            (the easiest binary framing of G3)

Reading of the outcome:
  - probe >> 1/k on G1/G2  => explicit commitment language is present and lexically
    detectable; the later failures cannot be dismissed as "no signal in the data".
  - probe ~ 1/k on G3/G4   => shallow lexical recoverability falls to the class floor
    at the action / collective-odds tier. That is what makes the difficulty gradient
    informative rather than an artifact.
Whichever way it comes out, we report it -- the two-sided design is the point.

TERMINOLOGY, do not drift (2026-07-26 audit): this is NOT a "signal ceiling" and NOT
a bound on what any model could achieve. A linear bag-of-words probe cannot rule out
non-linear, discourse-level, market-conditioned or user-history signal. It bounds
nothing; it measures shallow lexical recoverability.

The chronological split guarantees that all training comments precede the test
comments. It does NOT guarantee the absence of leakage: the same market and the same
commenter can appear on both sides, and the G3 label window can straddle the split
boundary. See e6b_group_splits.py for market-grouped and user-grouped splits.

Inputs : commitbench_v18/commitbench_v18/<model>/<platform>/<topic>/qwen3_*_task*.csv
Outputs: rebuttal/results/e6_signal_ceiling.csv, e6_summary.md
"""
import os, glob, sys, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = "commitbench_v18/commitbench_v18"
OUT  = "rebuttal/results"
os.makedirs(OUT, exist_ok=True)

# tier -> (filename suffix, label normaliser, valid gold labels).
# The valid-label filter drops a handful of malformed rows (blank / stray "True")
# that appear in a few released CSVs; without it they show up as extra classes.
TIERS = {
    "G1_has_position":  ("task0_0shot",
                         lambda s: s.astype(str).str.strip().str.lower()
                                    .map({"true": "HAS_POS", "false": "NO_POS"}),
                         {"HAS_POS", "NO_POS"}),
    "G2_position_side": ("task1",
                         lambda s: s.astype(str).str.strip().str.upper(),
                         {"YES", "NO"}),
    "G3_action_4way":   ("task2",
                         lambda s: s.astype(str).str.strip().str.lower(),
                         {"flip", "increase", "decrease", "hold"}),
    "G3b_flip_vs_hold": ("task2_flip_hold",
                         lambda s: s.astype(str).str.strip().str.lower(),
                         {"flip", "hold"}),
}


def norm_p(z):
    return math.erfc(abs(z) / math.sqrt(2))


KEY = ["market_id", "created_at", "text"]
EXCLUDED = {}   # (platform, tier) -> number of instance keys dropped for gold disagreement


def load_instances(suffix, platform, tag=None):
    """Pool the evaluated instances across models/topics and de-duplicate.

    Model coverage of the released instance pool is NOT identical, so this is the
    union of released evaluation instances, not one fixed set.

    AUDIT (2026-07-27): a small number of instances carry different gold labels in
    different model logs (E3b: 1.1% Polymarket / 2.5% Manifold on G3). De-duplicating
    on the key alone kept whichever label the file-glob happened to reach first, i.e.
    the probe's target depended on directory traversal order. Those keys are now
    dropped outright and counted, rather than silently resolved.
    """
    frames = []
    for f in glob.glob(f"{ROOT}/*/{platform}/*/*_{suffix}.csv"):
        try:
            d = pd.read_csv(f, dtype={"market_id": str, "created_at": str})
        except Exception:
            continue
        if d.empty or not {"gt", "text"}.issubset(d.columns):
            continue
        d["topic"] = os.path.basename(os.path.dirname(f))
        frames.append(d[["market_id", "created_at", "text", "gt", "topic"]])
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["text"] = d["text"].fillna("").astype(str)
    nun = d.assign(_g=d["gt"].astype(str).str.strip().str.lower()).groupby(KEY)["_g"].transform("nunique")
    n_drop = int(d.loc[nun > 1, KEY].drop_duplicates().shape[0])
    if tag is not None:
        EXCLUDED[tag] = (n_drop, int(d[KEY].drop_duplicates().shape[0]))
    d = d[nun == 1]
    d = d.drop_duplicates(subset=KEY).reset_index(drop=True)
    d = d[d.text.str.len() >= 10]
    d["_ts"] = pd.to_datetime(d["created_at"], utc=True, errors="coerce")
    return d.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)


def probe(d, tier, platform, n_perm=2000, seed=0):
    norm, valid = TIERS[tier][1], TIERS[tier][2]
    y = norm(d["gt"])
    keep = y.isin(valid)
    d, y = d[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    n = len(d)
    if n < 200 or y.nunique() < 2:
        return None
    cut = int(n * 0.7)                                   # time-split: train on the past
    Xtr_t, Xte_t = d.text.iloc[:cut], d.text.iloc[cut:]
    ytr, yte = y.iloc[:cut].values, y.iloc[cut:].values
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return None
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2,
                          sublinear_tf=True, stop_words="english")
    Xtr, Xte = vec.fit_transform(Xtr_t), vec.transform(Xte_t)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    clf.fit(Xtr, ytr)
    p = clf.predict(Xte)

    k_classes = len(valid)
    acc = accuracy_score(yte, p)
    bal = balanced_accuracy_score(yte, p)
    maj = pd.Series(ytr).value_counts().idxmax()
    acc_maj = accuracy_score(yte, [maj] * len(yte))

    # Primary signal test: balanced accuracy against the 1/k floor. Balanced accuracy
    # is immune to class imbalance (a constant predictor scores exactly 1/k), so this
    # asks "does the text carry ANY class-discriminative information" rather than
    # "can the probe exploit the label prior". Null = shuffled test labels, one-sided.
    rng = np.random.RandomState(seed)
    null = np.empty(n_perm)
    yte_arr = np.asarray(yte)
    for i in range(n_perm):
        null[i] = balanced_accuracy_score(rng.permutation(yte_arr), p)
    p_perm = float((null >= bal).sum() + 1) / (n_perm + 1)

    # Secondary: accuracy against the majority-class baseline (one-sided z).
    se = math.sqrt(max(acc_maj * (1 - acc_maj), 1e-12) / len(yte))
    z = (acc - acc_maj) / se
    p_maj = norm_p(z) / 2 if z > 0 else 1 - norm_p(z) / 2

    return dict(tier=tier, platform=platform, n=n, n_train=cut, n_test=len(yte),
                n_classes=k_classes, uniform_chance=1.0 / k_classes,
                acc_majority=acc_maj, acc_probe=acc, bal_acc_probe=bal,
                macro_f1_probe=f1_score(yte, p, average="macro"),
                bal_gain_over_chance=bal - 1.0 / k_classes, p_perm_bal_acc=p_perm,
                acc_gain_over_majority=acc - acc_maj, p_vs_majority=p_maj,
                _test_keys=set(zip(d.market_id.iloc[cut:], d.created_at.iloc[cut:])))


def llm_on_same_test(suffix, platform, tier, test_keys):
    """Score every evaluated LLM on the probe's held-out window.

    Each model is scored on the subset of that window present in its own released
    logs -- coverage differs by model -- so 'LLM above the probe' means the LLM
    extracts something a lexical model cannot, on that model's own coverage.
    """
    norm, valid = TIERS[tier][1], TIERS[tier][2]
    out = []
    for f in glob.glob(f"{ROOT}/*/{platform}/*/*_{suffix}.csv"):
        model = f.replace("\\", "/").split("/")[-4]
        try:
            d = pd.read_csv(f, dtype={"market_id": str, "created_at": str})
        except Exception:
            continue
        if d.empty or not {"gt", "pred"}.issubset(d.columns):
            continue
        d = d[[(m, c) in test_keys for m, c in zip(d.market_id, d.created_at)]]
        if d.empty:
            continue
        g, p = norm(d["gt"]), norm(d["pred"])
        keep = g.isin(valid)
        if keep.sum() == 0:
            continue
        out.append(pd.DataFrame(dict(model=model, gt=g[keep], pred=p[keep])))
    if not out:
        return pd.DataFrame()
    a = pd.concat(out, ignore_index=True)
    rows = []
    for model, s in a.groupby("model"):
        # unparseable / out-of-label predictions count as wrong, never as a class
        yt, yp = s["gt"], s["pred"]
        rows.append(dict(tier=tier, platform=platform, model=model, n=len(s),
                         acc=(yt == yp).mean(),
                         bal_acc=balanced_accuracy_score(
                             yt.values, yp.where(yp.isin(valid), "__OTHER__").values)))
    return pd.DataFrame(rows)


rows, llm_rows = [], []
for platform in ["manifold", "polymarket"]:
    for tier, (suffix, _norm, _valid) in TIERS.items():
        d = load_instances(suffix, platform, tag=(platform, tier))
        if d.empty:
            print(f"  [skip] {platform}/{tier}: no instances", flush=True)
            continue
        r = probe(d, tier, platform)
        show = ("-> " + str({k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in r.items()
                             if k in ("acc_probe", "bal_acc_probe", "acc_majority",
                                      "p_perm_bal_acc")})) if r else "-> too few"
        print(f"  {platform:11s} {tier:18s} n={len(d):6d} {show}", flush=True)
        if r:
            keys = r.pop("_test_keys")
            rows.append(r)
            lr = llm_on_same_test(suffix, platform, tier, keys)
            if not lr.empty:
                llm_rows.append(lr)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/e6_signal_ceiling.csv", index=False)
llm = pd.concat(llm_rows, ignore_index=True) if llm_rows else pd.DataFrame()
if not llm.empty:
    llm.to_csv(f"{OUT}/e6_llm_on_probe_testset.csv", index=False)

# ---------------------------------------------------------------- summary
L = ["# E6 - Supervised lexical recoverability probe, per task tier (P0-1)\n",
     "Concern (Fcai W1 / Meta c.1): prediction-market discourse is cheap talk, so the "
     "failures may reflect noisy data rather than model limitations.\n",
     "**Design.** A plain TF-IDF + logistic-regression probe, trained with a 70/30 "
     "chronological split on the union of released evaluation instances, so that all "
     "training comments precede the test comments. No LLM, no market features, comment "
     "text only. This measures how much of each task's label is recoverable from shallow "
     "lexical features.\n",
     "**Not a ceiling and not a bound.** A linear bag-of-words probe cannot rule out "
     "non-linear, discourse-level, market-conditioned or user-history signal, so a low "
     "score means low shallow lexical recoverability, nothing stronger. The chronological "
     "split also does not establish the absence of leakage: the same market and commenter "
     "can appear on both sides. See `e6b_group_splits.md` for grouped splits.\n",
     "**Gold consistency.** Instance keys whose gold label disagrees across model logs are "
     "excluded rather than resolved by traversal order: " +
     "; ".join(f"{p}/{t} {n} of {tot} keys" for (p, t), (n, tot) in EXCLUDED.items()) + ".\n",
     "Primary test = **balanced accuracy vs the 1/k floor** (a constant predictor scores "
     "exactly 1/k, so this is immune to class imbalance), significance by a 2000-draw "
     "label-permutation test. Secondary = raw accuracy vs the majority-class baseline.\n",
     "| Platform | Tier | n | k | Bal-Acc | 1/k | Bal gain | p(perm) | Probe Acc | Majority | Acc gain | Macro-F1 |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in res.iterrows():
    L.append(f"| {r.platform} | {r.tier} | {int(r.n)} | {int(r.n_classes)} | {r.bal_acc_probe:.3f} | "
             f"{r.uniform_chance:.3f} | {r.bal_gain_over_chance:+.3f} | {r.p_perm_bal_acc:.4f} | "
             f"{r.acc_probe:.3f} | {r.acc_majority:.3f} | {r.acc_gain_over_majority:+.3f} | "
             f"{r.macro_f1_probe:.3f} |")
L.append("")

if not res.empty:
    L.append("## Reading (per tier, per platform -- no averaging across platforms)")
    for tier in TIERS:
        sub = res[res.tier == tier]
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            has = r.bal_gain_over_chance > 0.02 and r.p_perm_bal_acc < 0.05
            L.append(f"- `{tier}` / {r.platform}: Bal-Acc {r.bal_acc_probe:.3f} vs floor "
                     f"{r.uniform_chance:.3f} ({r.bal_gain_over_chance:+.3f}, p={r.p_perm_bal_acc:.4f}) => "
                     + ("**recoverable signal in the text**" if has else
                        "**no signal recoverable above the class floor**"))
    if not llm.empty:
        L += ["",
              "## Probe vs the evaluated LLMs, on the probe's held-out window",
              "_Balanced accuracy. Each LLM is scored on the part of that window present in its "
              "own released logs, so coverage differs by model. 'LLM > probe' means the LLM "
              "extracts something a lexical model cannot._\n",
              "| Platform | Tier | n(test) | 1/k floor | Probe Bal-Acc | best LLM (Bal-Acc) | mean LLM | #LLMs > probe |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in res.iterrows():
            s = llm[(llm.tier == r.tier) & (llm.platform == r.platform)]
            if s.empty:
                continue
            b = s.loc[s.bal_acc.idxmax()]
            L.append(f"| {r.platform} | {r.tier} | {int(s.n.median())} | {r.uniform_chance:.3f} | "
                     f"{r.bal_acc_probe:.3f} | {b.model} ({b.bal_acc:.3f}) | {s.bal_acc.mean():.3f} | "
                     f"{int((s.bal_acc > r.bal_acc_probe).sum())}/{len(s)} |")
    L += ["",
          "## Caveat that must travel with the G1 row",
          "`common.run_task0` builds the G1 positive pool as positioned comments that pass "
          "`has_explicit_pos_signal` (and `_score_explicit_signal >= EXPLICIT_QUALITY_THR`), "
          "while the negative pool is un-positioned comments with no such filter. G1 is therefore "
          "*by construction* a detect-explicit-commitment-language task, and a lexical probe "
          "partly recovers that selection rule. Read the G1 numbers as 'explicit commitment "
          "language is reliably present and reliably detectable', which is what the paper claims "
          "-- not as 'latent commitment is recoverable from arbitrary text'. The G2/G3 rows carry "
          "no such construction advantage, which is what makes the drop across tiers meaningful.",
          "",
          "The tiers are ordered by how far the label sits from the speaker's own utterance: "
          "own commitment (G1/G2) -> own future action (G3) -> the crowd's price move (G4, see E4). "
          "The gradient across tiers is the quantity Fcai's W1 asks about, within the limits of a "
          "lexical probe: it separates 'not recoverable from the words alone' from 'recoverable "
          "and the models still miss it'.",
          "",
          "**Use in the rebuttal.** Pair this with E1 (Wrong-Bettor, reported with its BH-FDR "
          "correction and macro mean of 0.493, not the peak alone) and E4 (a full-window lexical "
          "probe on G4 windows). Report the tiers that show recoverable signal as evidence against "
          "'cheap talk only', and the tiers that do not as a quantified interpretation boundary. "
          "Do not describe any of it as a ceiling, a bound, or leak-free."]

open(f"{OUT}/e6_summary.md", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
print(f"\n[written] e6_signal_ceiling.csv, e6_summary.md")
