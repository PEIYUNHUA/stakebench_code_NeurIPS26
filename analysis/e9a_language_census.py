"""
E9a: proper language identification over the comments the English filter excludes,
and a census of how many of them can carry a G1/G2 gold label.

Why this exists
---------------
E7 classified the excluded comments with a function-word vote and left 17,267 of
23,221 unclassified -- exactly the kind of homemade heuristic the audit told us not
to lean on. This replaces it with py3langid (a standard LID model) and answers the
only question that gates E9: for each language, how many excluded comments carry a
reconstructed position, i.e. can be given G1/G2 gold from market records alone?

Encoding note (matters a lot here)
----------------------------------
`common.load_data` reads the Polymarket export as latin-1, so genuinely UTF-8 text
(Romanian diacritics, Spanish tildes) reaches the benchmark as mojibake. LID on the
mojibake view would be meaningless, so we round-trip the bytes back to UTF-8 before
classifying. Both views are counted: the round-tripped one is the real comment, the
latin-1 one is what the released pipeline actually saw. The gap between them is
itself a finding for TPc3 Q2.

No LLM calls. Output: rebuttal/results/e9a_language_census.md
                      rebuttal/results/e9a_nonenglish_pool.csv   (candidate pool for E9b)
"""
import os, sys
import numpy as np
import pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

sys.path.insert(0, "stakebench_code_NeurIPS26")
os.environ.setdefault("COMMITBENCH_MANIFOLD_DIR",   os.path.abspath("data/manifold_data"))
os.environ.setdefault("COMMITBENCH_POLYMARKET_DIR", os.path.abspath("data/polymarket_data"))
import common
from py3langid.langid import LanguageIdentifier, MODEL_FILE
# norm_probs=True makes classify() return a probability in [0,1]; the default returns
# an unnormalised score that cannot be thresholded.
langid = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)

OUT = "rebuttal/results"
os.makedirs(OUT, exist_ok=True)
SRC = "data/polymarket_data/comments_clean.csv"
CHUNK = 100_000
MIN_LID_CHARS = 30          # LID on very short text is unreliable; below this we abstain
COLS = ["market_id", "question", "created_at", "text", "user_address",
        "position_side", "position_size"]


def to_utf8(s):
    """Undo the latin-1 read so Romanian/Spanish diacritics come back."""
    try:
        return s.encode("latin-1", errors="strict").decode("utf-8", errors="strict")
    except Exception:
        return s          # genuinely latin-1 or already clean ASCII


CACHE = f"{OUT}/e9a_nonenglish_all.csv"
STATS = f"{OUT}/e9a_stream_stats.csv"
if os.path.exists(CACHE):
    print(f"[1] reusing cached non-English pool {CACHE}")
    ne = pd.read_csv(CACHE, dtype=str, encoding="utf-8")
    ne["text_utf8"] = ne["text_utf8"].fillna("").astype(str)
    s = pd.read_csv(STATS).iloc[0]
    n_rows, n_quality, n_english, n_mojibake = int(s.n_rows), int(s.n_quality), int(s.n_english), int(s.n_mojibake)
else:
    print("[1] streaming Polymarket comments (latin-1 view, round-tripped to UTF-8) ...")
    kept, n_rows, n_quality, n_english, n_mojibake = [], 0, 0, 0, 0
    reader = pd.read_csv(SRC, usecols=lambda c: c.strip("﻿") in COLS, dtype=str,
                         encoding="latin-1", engine="c", chunksize=CHUNK, on_bad_lines="skip")
    for ci, ch in enumerate(reader):
        ch.columns = [c.strip("﻿") for c in ch.columns]
        ch["text"] = ch["text"].fillna("").astype(str)
        n_rows += len(ch)
        q = ch[ch["text"].apply(common.is_quality_text)]
        n_quality += len(q)
        en_mask = q["text"].apply(common.is_english)
        n_english += int(en_mask.sum())
        sub = q[~en_mask].copy()
        if len(sub):
            sub["text_utf8"] = sub["text"].map(to_utf8)
            n_mojibake += int((sub["text_utf8"] != sub["text"]).sum())
            kept.append(sub)
        if (ci + 1) % 2 == 0:
            print(f"    {n_rows:,} rows | quality {n_quality:,} | english {n_english:,} "
                  f"| non-english {sum(len(k) for k in kept):,}", flush=True)
    ne = pd.concat(kept, ignore_index=True)
    ne.to_csv(CACHE, index=False, encoding="utf-8")
    pd.DataFrame([dict(n_rows=n_rows, n_quality=n_quality, n_english=n_english,
                       n_mojibake=n_mojibake)]).to_csv(STATS, index=False)
print(f"    {n_rows:,} rows | quality {n_quality:,} | english {n_english:,} "
      f"| non-english {len(ne):,} | recovered from mojibake {n_mojibake:,}")

print("[2] language identification (py3langid) ...")
lang, conf = [], []
for t in ne["text_utf8"].values:
    if len(t) < MIN_LID_CHARS:
        lang.append("_short"); conf.append(np.nan); continue
    l, c = langid.classify(t)
    lang.append(l); conf.append(float(c))
ne["lang"], ne["lang_conf"] = lang, conf

# py3langid returns a normalised probability in [0,1] for 0.3.x; guard either way
if np.nanmax(ne["lang_conf"].values) > 1.0:
    ne["lang_conf"] = np.nan          # raw log-prob: not comparable, drop rather than mislead
    CONF_NOTE = ("py3langid returned unnormalised scores on this install, so no confidence "
                 "threshold is applied; treat the per-language counts as approximate.")
    hi = ne
else:
    CONF_NOTE = "High-confidence = py3langid probability >= 0.90."
    hi = ne[ne["lang_conf"] >= 0.90]

print("[3] gold availability per language ...")
# The raw export stores "Yes"/"No"; common.load_data upper-cases before the isin check,
# so we must do the same or every row looks unpositioned.
ne["position_side"] = ne["position_side"].fillna("").astype(str).str.strip().str.upper()
ne["position_size"] = pd.to_numeric(ne["position_size"], errors="coerce")
ne["has_pos"] = ne["position_side"].isin(["YES", "NO"]) & (ne["position_size"].fillna(0) > 0)
ne["no_pos"] = ~ne["position_side"].isin(["YES", "NO"])

rows = []
for lg, g in ne.groupby("lang"):
    gh = g[g.lang_conf >= 0.90] if ne["lang_conf"].notna().any() else g
    rows.append(dict(lang=lg, n=len(g), n_highconf=len(gh),
                     n_pos=int(g.has_pos.sum()), n_neg=int(g.no_pos.sum()),
                     n_markets=g.market_id.nunique(),
                     median_len=int(g.text_utf8.str.len().median()),
                     n_mojibake=int((g.text_utf8 != g.text).sum())))
C = pd.DataFrame(rows).sort_values("n", ascending=False)

# candidate pool for E9b: the two languages the design needs
pool = ne[ne.lang.isin(["ro", "es"])].copy()
pool.to_csv(f"{OUT}/e9a_nonenglish_pool.csv", index=False, encoding="utf-8")

L = ["# E9a - language identification over the English-filter exclusions\n",
     f"Source: `{SRC}` ({n_rows:,} rows). Quality-passing: **{n_quality:,}**; "
     f"passing `is_english`: **{n_english:,}**; excluded as non-English: **{len(ne):,}**.\n",
     f"{n_mojibake:,} of the excluded comments contain bytes that only decode correctly as "
     "UTF-8, i.e. the released pipeline's latin-1 read shows them as mojibake. LID here runs "
     "on the round-tripped text.\n",
     f"LID: py3langid, abstaining below {MIN_LID_CHARS} characters (`_short`). {CONF_NOTE}\n",
     "## Per-language census\n",
     "| lang | n | high-conf | with position (G1 pos / G2 gold) | without position (G1 neg) | markets | median chars | mojibake |",
     "|---|---:|---:|---:|---:|---:|---:|---:|"]
for _, r in C.head(18).iterrows():
    L.append(f"| {r.lang} | {r.n:,} | {r.n_highconf:,} | {r.n_pos:,} | {r.n_neg:,} | "
             f"{r.n_markets:,} | {r.median_len} | {r.n_mojibake:,} |")
L.append("")
L.append(f"Wrote the Romanian + Spanish candidate pool to `e9a_nonenglish_pool.csv` "
         f"({len(pool):,} rows).\n")

L.append("## Feasibility gate for E9\n")
for lg in ["ro", "es"]:
    s = C[C.lang == lg]
    if s.empty:
        L.append(f"- **{lg}**: not detected -- this arm cannot run."); continue
    r = s.iloc[0]
    ok_g1 = min(r.n_pos, r.n_neg) >= 50
    L.append(f"- **{lg}**: {r.n_pos:,} positioned / {r.n_neg:,} unpositioned across "
             f"{r.n_markets:,} markets -> G1 balanced sample of 50+50 is "
             f"{'FEASIBLE' if ok_g1 else 'NOT feasible'}; G2 (side labels) draws from the "
             f"{r.n_pos:,} positioned rows.")
L.append("")
L.append("Note: the non-English G1 pool must use a **position-only** construction. The paper's "
         "positive pool is filtered by `has_explicit_pos_signal`, an English lexicon, which "
         "selects almost nothing in Romanian or Spanish. Levels are therefore comparable only "
         "*within* an instance across languages, never across the two instance pools.")

open(f"{OUT}/e9a_language_census.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
print(f"\n-> wrote {OUT}/e9a_language_census.md and e9a_nonenglish_pool.csv")
