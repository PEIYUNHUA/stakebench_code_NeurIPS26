"""
E9c: which commitment cue families survive across languages, measured without any LLM.

Reviewer concern (TPc3 Q2, second half): how do language-specific writing styles affect
commitment cues? This script answers the part that needs no new inference. It splits the
released corpus by language with offline language identification and measures how often
each cue family appears in English against non-English comments.

The hypothesis being tested is that commitment cues split into two kinds:
  quantity cues   -- amounts, currency symbols, percentages, prices. Language-independent.
  lexical cues    -- first-person holding verbs, trade verbs, side words, certainty words.
                     Carried by an English lexicon, so they should not survive translation
                     of the discourse into another language.

Main population is positioned comments (position_size > 0), because those are the ones that
carry a verified commitment. An all-comments version is reported alongside.

Inputs:
  data/polymarket_data/comments_clean.csv
  data/manifold_data/comments_clean.csv

Outputs (rebuttal/results/e9/):
  e9c_cue_portability.csv    cue family x language group, hit rate, Cohen's h
  e9c_language_counts.csv    language distribution after quality filter
  e9c_summary.md
"""
import os, re, sys, json
import numpy as np
import pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

OUT = "rebuttal/results/e9"
os.makedirs(OUT, exist_ok=True)
LID_CONF = 0.90

# ── quality filter mirrors stakebench_code_NeurIPS26/common.py:753 with MIN_TEXT_LEN = 30
MIN_TEXT_LEN = 30
_URL_RE = re.compile(r"https?://\S+|www\.\S+")

def is_quality_text(t):
    if not isinstance(t, str): return False
    c = _URL_RE.sub("", t).strip()
    if len(c) < MIN_TEXT_LEN: return False
    return sum(ch.isalpha() for ch in c) / max(len(c), 1) >= 0.3

# ── cue families ────────────────────────────────────────────────────────────
CUES = {
    "money amount":      re.compile(r"[$€£¥]\s*\d|\d[\d,.]*\s*(?:dollar|buck|usd|mana|m\$)", re.I),
    "currency symbol":   re.compile(r"[$€£¥]|\bm\$"),
    "percentage":        re.compile(r"\d\s*%|\b\d{1,3}\s*percent\b", re.I),
    "price or odds":     re.compile(r"\b0?\.\d{1,3}\b|\b\d{1,3}\s*(?:c|cents?)\b", re.I),
    "first-person hold": re.compile(r"\bi\s+(?:have\s+)?(?:bought|sold|hold|holding|own|added|entered|exited|cut|went)\b", re.I),
    "trade verb":        re.compile(r"\b(?:buy|bought|sell|sold|long|short|hedge|hedging|trim|trimming|dump|dumped|load|loading)\b", re.I),
    "side word":         re.compile(r"\b(?:yes|no)\b", re.I),
    "certainty marker":  re.compile(r"\b(?:sure|certain|definitely|obviously|clearly|guaranteed|no doubt)\b", re.I),
}
QUANTITY = {"money amount", "currency symbol", "percentage", "price or odds"}


def cohens_h(p1, p2):
    return 2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p2))


def pick_encoding(path):
    """Non-English characters must survive on disk or the language split is meaningless."""
    best = None
    for enc in ("utf-8", "latin-1"):
        try:
            d = pd.read_csv(path, nrows=200000, usecols=["text"], encoding=enc)
        except Exception:
            continue
        s = d.text.astype(str)
        moji = s.str.contains(r"Ã|â€|Å¾|Ä‚", regex=True).mean()
        dia  = s.str.contains(r"[ăâîșțáéíóúñçüöğışİ]", case=False, regex=True).mean()
        print(f"    {enc}: mojibake={moji:.4f} diacritics={dia:.4f}")
        cand = (moji, -dia, enc)
        if best is None or cand < best: best = cand
    return best[2], {"mojibake": round(best[0], 5), "diacritics": round(-best[1], 5)}


def load(platform):
    path = f"data/{platform}_data/comments_clean.csv"
    print(f"  [{platform}] encoding check")
    enc, diag = pick_encoding(path)
    print(f"  [{platform}] using {enc}")
    cols = ["market_id", "comment_id", "text", "position_size"]
    d = pd.read_csv(path, low_memory=False, encoding=enc, usecols=cols)
    d["platform"] = platform
    d["sz"] = pd.to_numeric(d.position_size, errors="coerce").fillna(0)
    d = d[d.text.map(is_quality_text)].copy()
    print(f"  [{platform}] quality-passing comments: {len(d):,}")
    return d, {"encoding": enc, **diag, "quality_passing": int(len(d))}


def identify(texts):
    import py3langid
    try:
        from py3langid.langid import LanguageIdentifier, MODEL_FILE
        ident = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
        cls = ident.classify
    except Exception:
        cls = py3langid.classify          # falls back to log probabilities
    langs, confs = [], []
    for t in texts:
        try:
            lg, cf = cls(str(t)[:1000])
        except Exception:
            lg, cf = "un", 0.0
        langs.append(lg); confs.append(float(cf))
    return langs, confs


def rates(df):
    return {k: float(df.text.str.contains(rx, regex=True).mean()) for k, rx in CUES.items()}


def main():
    meta = {}
    frames = []
    for p in ("polymarket", "manifold"):
        d, m = load(p); frames.append(d); meta[p] = m
    df = pd.concat(frames, ignore_index=True)

    print(f"\n[LID] classifying {len(df):,} comments with py3langid ...")
    df["lang"], df["lang_conf"] = identify(df.text.tolist())
    conf_ok = df.lang_conf >= LID_CONF
    print(f"  high-confidence (>= {LID_CONF}): {conf_ok.mean():.1%}")

    lc = (df[conf_ok].groupby("lang").size().sort_values(ascending=False)
            .rename("n").reset_index())
    lc["share"] = (lc.n / lc.n.sum()).round(5)
    lc.to_csv(f"{OUT}/e9c_language_counts.csv", index=False)
    print("  top languages:", ", ".join(f"{r.lang}={r.n:,}" for r in lc.head(8).itertuples()))

    rows = []
    for pop_name, pop in [("positioned", df[df.sz > 0]), ("all", df)]:
        pop = pop[pop.lang_conf >= LID_CONF]
        en  = pop[pop.lang == "en"]
        non = pop[pop.lang != "en"]
        if len(en) == 0 or len(non) == 0: continue
        r_en, r_non = rates(en), rates(non)
        for cue in CUES:
            rows.append(dict(population=pop_name, cue=cue,
                             family="quantity" if cue in QUANTITY else "lexical",
                             n_english=len(en), n_non_english=len(non),
                             rate_english=round(r_en[cue], 4),
                             rate_non_english=round(r_non[cue], 4),
                             ratio=round(r_non[cue] / r_en[cue], 3) if r_en[cue] > 0 else np.nan,
                             cohens_h=round(float(cohens_h(r_non[cue], r_en[cue])), 3)))
        # per-language detail for the largest non-English languages
        for lg in [x for x in lc.lang.head(8) if x != "en"]:
            sub = pop[pop.lang == lg]
            if len(sub) < 100: continue
            r = rates(sub)
            for cue in CUES:
                rows.append(dict(population=pop_name, cue=cue,
                                 family="quantity" if cue in QUANTITY else "lexical",
                                 n_english=len(en), n_non_english=len(sub),
                                 rate_english=round(r_en[cue], 4),
                                 rate_non_english=round(r[cue], 4),
                                 ratio=round(r[cue] / r_en[cue], 3) if r_en[cue] > 0 else np.nan,
                                 cohens_h=round(float(cohens_h(r[cue], r_en[cue])), 3),
                                 language=lg))
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/e9c_cue_portability.csv", index=False)

    main_tbl = out[(out.population == "positioned") & (out.get("language").isna())] \
               if "language" in out.columns else out[out.population == "positioned"]
    print("\n=== positioned comments, English vs all non-English ===")
    print(main_tbl[["cue", "family", "rate_english", "rate_non_english", "ratio", "cohens_h"]]
          .to_string(index=False))

    q = main_tbl[main_tbl.family == "quantity"]
    l = main_tbl[main_tbl.family == "lexical"]
    with open(f"{OUT}/e9c_summary.md", "w", encoding="utf-8") as fh:
        fh.write("# E9c — cue-family portability across languages (no LLM)\n\n")
        fh.write(f"Offline language identification (py3langid, confidence >= {LID_CONF}) over "
                 f"quality-passing comments from both platforms.\n\n")
        fh.write("## Positioned comments, English against all non-English\n\n")
        fh.write(main_tbl[["cue","family","n_english","n_non_english","rate_english",
                           "rate_non_english","ratio","cohens_h"]].to_markdown(index=False) + "\n\n")
        fh.write(f"Mean retention ratio, quantity cues: {q.ratio.mean():.2f}. "
                 f"Lexical cues: {l.ratio.mean():.2f}.\n\n")
        fh.write("Language identification is a classifier and carries its own error rate, so "
                 "these groups are high-confidence language assignments rather than verified "
                 "labels. Rates are descriptive and no model was run.\n")
        fh.write("\n## Encoding and filter diagnostics\n\n```\n" + json.dumps(meta, indent=2) + "\n```\n")

    print(f"\nMean retention ratio: quantity cues {q.ratio.mean():.2f}   lexical cues {l.ratio.mean():.2f}")
    print(f"wrote {OUT}/e9c_cue_portability.csv, e9c_language_counts.csv, e9c_summary.md")


if __name__ == "__main__":
    main()
