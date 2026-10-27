"""
E8 (K2h4 W1 / Meta d.3, no new LLM calls): HOW does market commitment show up in
language? Contrast the wording of comments by users who hold a position against
comments by users who do not.

K2h4: "some examples highlighting the influence of financial commitments on language
would be useful". Meta (d.3) asks for 5-10 qualitative examples "contrasting with
non-committed language". E3 already pulls per-task failure cases; this script supplies
the *systematic* half -- what differs lexically -- so the hand-picked examples in E3
illustrate a measured ASSOCIATION rather than anecdotes.

WORDING RULES (2026-07-26 / 2026-07-27 audits) -- the G1 positive pool is selected by an
explicit-position lexicon, so every number here is descriptive and partly circular:
  * association, never influence / effect / causes;
  * same-market illustrative contrasts, never matched pairs (nothing is matched on time,
    length, activity or thread depth);
  * markers outside the selection lexicon give ADDITIONAL DESCRIPTIVE CONTRASTS -- they
    are not "independent of construction", because the sampling rule can shift any
    marker through correlated features;
  * the logistic terms are discriminative terms IN THE CONSTRUCTED EVALUATION SET: the
    classifier is fitted on pooled G1 data with no held-out validation and no market,
    topic or user controls.

Two views:
  1. Discriminative terms: TF-IDF + logistic regression over G1 instances
     (gt = commenter holds a position vs not), reporting the highest-weight terms
     on each side. These are the cues the benchmark's G1 signal (E6: bal-acc
     0.685/0.729) is built from.
  2. Interpretable markers: hand-specified lexical families (first-person holding
     verbs, money amounts, side words, hedges, certainty, questions) with rates in
     each class and Cohen's h effect sizes.
Plus matched example pairs from the same market, one committed and one not.

Inputs : commitbench_v18/commitbench_v18/<model>/<platform>/<topic>/*_task0_0shot.csv
Outputs: rebuttal/results/e8_terms.csv, e8_markers.csv, e8_summary.md
"""
import os, glob, sys, re, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = "commitbench_v18/commitbench_v18"
OUT  = "rebuttal/results"
os.makedirs(OUT, exist_ok=True)

MARKERS = {
    "first_person_holding": r"\b(i|i'm|im|i've|ive)\s+(bought|sold|bet|hold|holding|own|long|short|in|out|added|dumped)\b|\bmy\s+(position|bet|shares|stake)\b",
    "money_amount":         r"\$\s*\d|\b\d[\d,]*\s*(?:k|dollar|buck|share|shares)\b",
    "side_word":            r"\b(yes|no)\s*(side|shares|position)?\b",
    "trade_verb":           r"\b(buy|buying|sell|selling|bought|sold|bet|betting|long|short|hedge|hedged)\b",
    "certainty":            r"\b(definitely|certainly|obviously|clearly|guaranteed|for sure|no way|100%)\b",
    "hedge":                r"\b(maybe|probably|might|could|perhaps|seems|i think|not sure|unsure|likely)\b",
    "question":             r"\?",
    "price_talk":           r"\b\d{1,2}\s*%|\bodds?\b|\bprice\b|\bcents?\b",
}
MARKER_RE = {k: re.compile(v, re.I) for k, v in MARKERS.items()}


def cohens_h(p1, p2):
    """Effect size for a difference of two proportions."""
    f = lambda p: 2 * math.asin(math.sqrt(min(max(p, 0.0), 1.0)))
    return f(p1) - f(p2)


def load_g1(platform):
    frames = []
    for f in glob.glob(f"{ROOT}/*/{platform}/*/*_task0_0shot.csv"):
        try:
            d = pd.read_csv(f, dtype={"market_id": str, "created_at": str})
        except Exception:
            continue
        if d.empty or not {"gt", "text"}.issubset(d.columns):
            continue
        d["topic"] = os.path.basename(os.path.dirname(f))
        frames.append(d[["market_id", "created_at", "question", "text", "gt", "topic"]])
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = d.drop_duplicates(subset=["market_id", "created_at", "text"]).reset_index(drop=True)
    d["text"] = d["text"].fillna("").astype(str)
    d["held"] = d["gt"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
    return d[d.held.notna() & (d.text.str.len() >= 10)].reset_index(drop=True)


term_rows, marker_rows, example_blocks = [], [], []
for platform in ["manifold", "polymarket"]:
    d = load_g1(platform)
    if d.empty or d.held.nunique() < 2:
        print(f"[skip] {platform}", flush=True)
        continue
    print(f"[{platform}] n={len(d)}  held={int(d.held.sum())}  not-held={int((~d.held).sum())}", flush=True)

    # ---- 1. discriminative terms
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3,
                          sublinear_tf=True, stop_words="english")
    X = vec.fit_transform(d.text)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X, d.held.values)
    names = np.array(vec.get_feature_names_out())
    coef = clf.coef_[0]                       # positive weight -> predicts HELD
    order = np.argsort(coef)
    for idx, side in [(order[::-1][:25], "committed"), (order[:25], "non_committed")]:
        for i in idx:
            term_rows.append(dict(platform=platform, side=side, term=names[i],
                                  weight=float(coef[i])))

    # ---- 2. interpretable markers
    for name, rx in MARKER_RE.items():
        hit = d.text.str.contains(rx, regex=True)
        p_held = float(hit[d.held].mean()); p_not = float(hit[~d.held].mean())
        marker_rows.append(dict(platform=platform, marker=name,
                                rate_held=p_held, rate_not_held=p_not,
                                diff=p_held - p_not, cohens_h=cohens_h(p_held, p_not),
                                n_held=int(d.held.sum()), n_not=int((~d.held).sum())))

    # ---- 3. illustrative same-market contrasts (NOT a matched design: first held and
    #         first unheld comment in the market, nothing matched on time/length/user)
    got = 0
    for mid, g in d.groupby("market_id"):
        if got >= 3 or g.held.nunique() < 2:
            continue
        a = g[g.held].iloc[0]; b = g[~g.held].iloc[0]
        example_blocks.append(
            f"**{platform} / {a.topic}** - market: _{str(a.question)[:110]}_\n"
            f"- HOLDS a position: \"{a.text[:200]}\"\n"
            f"- NO position:      \"{b.text[:200]}\"")
        got += 1

terms = pd.DataFrame(term_rows);  terms.to_csv(f"{OUT}/e8_terms.csv", index=False)
mark  = pd.DataFrame(marker_rows); mark.to_csv(f"{OUT}/e8_markers.csv", index=False)

L = ["# E8 - How commitment shows up in language: committed vs non-committed contrast (K2h4, Meta d.3)\n",
     "Systematic counterpart to the hand-picked cases in `e3_examples.md`. Built on the G1 "
     "instance set (gold label = the commenter verifiably held a position in that market), "
     "pooled across topics, de-duplicated.\n",
     "## Interpretable markers (rate among commenters who HOLD vs who do NOT)"]
if not mark.empty:
    L += ["| Platform | Marker | held | not held | diff | Cohen's h |", "|---|---|---|---|---|---|"]
    for _, r in mark.sort_values(["platform", "cohens_h"], ascending=[True, False]).iterrows():
        L.append(f"| {r.platform} | `{r.marker}` | {r.rate_held:.3f} | {r.rate_not_held:.3f} | "
                 f"{r['diff']:+.3f} | {r.cohens_h:+.3f} |")
    L.append("")
    top = mark.reindex(mark.cohens_h.abs().sort_values(ascending=False).index).head(4)
    L.append("- Largest separations: " + "; ".join(
        f"`{r.marker}` ({r.platform}, h={r.cohens_h:+.2f})" for _, r in top.iterrows()) + ".")
    L.append("")

if not terms.empty:
    L.append("## Most discriminative terms in the constructed evaluation set "
             "(TF-IDF + logistic regression over pooled G1; no held-out validation, "
             "no market / topic / user controls)")
    for platform in terms.platform.unique():
        for side, label in [("committed", "predicts a HELD position"),
                            ("non_committed", "predicts NO position")]:
            t = terms[(terms.platform == platform) & (terms.side == side)].head(20)
            L.append(f"- **{platform} / {label}**: " + ", ".join(f"`{x}`" for x in t.term))
    L.append("")

if example_blocks:
    L += ["## Same-market illustrative contrasts (position-holder vs no position)",
          "_Not a matched design: the first held and the first unheld comment in each market, "
          "with no matching on time, length, commenter activity or thread depth._", ""]
    L += [b + "\n" for b in example_blocks]

L += ["## Caveat: part of this contrast is built in, and the rebuttal must say so",
      "`common.run_task0` selects the G1 positive pool with `has_explicit_pos_signal` + "
      "`_score_explicit_signal`, whose components are money amounts, first-person holding verbs "
      "and side words. The `money_amount` / `first_person_holding` / `side_word` rows above "
      "therefore partly re-measure the sampling rule rather than discovering it. Use them to "
      "*describe* what explicit commitment language looks like (which is what K2h4 asked for). "
      "The markers NOT in the selection lexicon -- `certainty`, `hedge`, `question`, `price_talk` "
      "-- provide ADDITIONAL DESCRIPTIVE CONTRASTS. They are not independent of the construction: "
      "a sampling rule can shift any marker through features correlated with the lexicon.",
      "",
      "## Reading",
      "- This gives the rebuttal a measured statement rather than an anecdote: explicit commitment "
      "language is lexically visible (E6 puts G1 lexical recoverability at bal-acc 0.685 Manifold / "
      "0.729 Polymarket), and the markers above name *what* is visible.",
      "- Pair with `e3b_verified_examples.md` (NOT `e3_examples.md`, which is unverified candidate "
      "output) for the 8-10 cases the AC asked for in item d.3, using the marker table as the "
      "framing sentence.",
      "- Platform scope: the certainty/question pattern is a Manifold observation. Check the "
      "per-platform rows before generalising it -- the Polymarket question difference is near zero, "
      "so 'commitment surfaces as specificity, not confidence' must be stated for Manifold only.",
      "- These are associations within the constructed evaluation set, not a causal claim about how "
      "holding a position changes writing. The within-user DiD in the paper is the causal-flavoured "
      "evidence; this is the descriptive layer."]

open(f"{OUT}/e8_summary.md", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
print(f"\n[written] e8_terms.csv, e8_markers.csv, e8_summary.md")
