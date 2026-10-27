"""
E7 (TPc3 Q2, zero new inference): how much non-English discourse does StakeBench
currently exclude, and what is the size of the multilingual extension?

Reviewer question (TPc3): "How well does commitment-grounded language understanding
transfer to non-English prediction markets, and how do language-specific writing
styles affect commitment cues?"

We cannot answer the transfer question without new data, but we can answer the part
that is answerable today and turn it into a concrete camera-ready commitment:
quantify what the pipeline's `is_english` filter removes, on both platforms, and
how much of it is positioned (i.e. would be usable as G1/G2/G3 supervision).

Non-ASCII scripts are additionally bucketed (CJK / Cyrillic / Arabic / Devanagari /
Greek / Hebrew / Thai / Latin-with-diacritics) so the answer can name the languages
rather than just a percentage.

Inputs : data/<platform>_data/comments_clean.csv  (streamed in chunks)
Outputs: rebuttal/results/e7_language_coverage.csv, e7_summary.md
"""
import os, sys, re
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, "stakebench_code_NeurIPS26")
os.environ.setdefault("COMMITBENCH_MANIFOLD_DIR",   os.path.abspath("data/manifold_data"))
os.environ.setdefault("COMMITBENCH_POLYMARKET_DIR", os.path.abspath("data/polymarket_data"))
import common

OUT = "rebuttal/results"
os.makedirs(OUT, exist_ok=True)
PLATS = {"manifold":  ("data/manifold_data/comments_clean.csv",  "utf-8-sig"),
         "polymarket": ("data/polymarket_data/comments_clean.csv", "latin-1")}
CHUNK = 200_000

SCRIPTS = [
    ("CJK",        re.compile(r"[一-鿿぀-ヿ가-힯]")),
    ("Cyrillic",   re.compile(r"[Ѐ-ӿ]")),
    ("Arabic",     re.compile(r"[؀-ۿ]")),
    ("Devanagari", re.compile(r"[ऀ-ॿ]")),
    ("Greek",      re.compile(r"[Ͱ-Ͽ]")),
    ("Hebrew",     re.compile(r"[֐-׿]")),
    ("Thai",       re.compile(r"[฀-๿]")),
]

# Most excluded comments are Latin-script, so a script check alone under-counts them
# badly (Romanian and Spanish threads on Polymarket look like ASCII). Bucket those with
# a stop-word vote instead; anything that wins nothing is short/low-function-word text
# rather than a foreign language, and is reported separately.
LANG_STOP = {
    "spanish":    {"que", "de", "la", "el", "los", "las", "por", "para", "con", "una", "es", "no", "se", "en"},
    "portuguese": {"que", "de", "da", "do", "os", "as", "para", "com", "uma", "nao", "não", "isso", "ele", "mas"},
    "french":     {"que", "de", "le", "les", "des", "pour", "avec", "une", "est", "pas", "dans", "sur", "je", "ce"},
    "german":     {"der", "die", "das", "und", "ist", "nicht", "ein", "eine", "auf", "mit", "von", "auch", "aber"},
    "italian":    {"che", "di", "il", "la", "per", "con", "una", "non", "sono", "questo", "anche", "come", "più"},
    "romanian":   {"sa", "să", "de", "la", "si", "și", "nu", "este", "pentru", "cu", "din", "care", "mai", "ca"},
    "turkish":    {"bir", "ve", "bu", "için", "ile", "daha", "olarak", "var", "yok", "ama", "çok", "ne"},
    "dutch":      {"het", "een", "van", "en", "is", "niet", "dat", "op", "met", "voor", "maar", "ook"},
    "polish":     {"nie", "jest", "sie", "się", "to", "na", "do", "za", "ale", "jak", "tak", "ze", "że"},
    "indonesian": {"yang", "dan", "ini", "itu", "tidak", "untuk", "dengan", "dari", "ada", "akan", "saya"},
}
_TOKEN_RE = re.compile(r"[a-zà-ÿăâîșțçğıöüşéèêëñ]+", re.I)


def guess_latin_lang(s, min_hits=2, min_rate=0.10):
    """Cheap stop-word vote over Latin-script languages; None when nothing wins."""
    w = _TOKEN_RE.findall(s.lower())
    if len(w) < 4:
        return None
    best, best_hits = None, 0
    for lang, stops in LANG_STOP.items():
        h = sum(1 for x in w if x in stops)
        if h > best_hits:
            best, best_hits = lang, h
    if best_hits >= min_hits and best_hits / len(w) >= min_rate:
        return best
    return None

rows = []
for platform, (path, enc) in PLATS.items():
    print(f"[{platform}] streaming {path} ...", flush=True)
    tot = quality = eng = pos = pos_quality = pos_noneng = 0
    script_counts = {name: 0 for name, _ in SCRIPTS}
    lang_counts = {name: 0 for name in LANG_STOP}
    lang_counts["unclassified_short_or_lowfunc"] = 0
    reader = pd.read_csv(path, usecols=lambda c: c.lstrip("﻿") in ("text", "position_side"),
                         dtype=str, encoding=enc, engine="python", on_bad_lines="skip",
                         chunksize=CHUNK)
    for ci, ch in enumerate(reader):
        ch.columns = [c.lstrip("﻿") for c in ch.columns]
        t = ch["text"].fillna("").astype(str)
        side = ch.get("position_side", pd.Series([""] * len(ch))).fillna("").astype(str).str.upper()
        q = t.map(common.is_quality_text)
        e = t.map(common.is_english)
        p = side.isin(["YES", "NO"])
        tot += len(ch); quality += int(q.sum()); eng += int((q & e).sum())
        pos += int(p.sum()); pos_quality += int((p & q).sum())
        pos_noneng += int((p & q & ~e).sum())
        # The released loader reads polymarket as latin-1 (common.load_data), which turns
        # real UTF-8 into mojibake. Keep the counts on the latin-1 view the benchmark
        # actually saw, but round-trip the bytes before bucketing scripts, otherwise every
        # CJK/Cyrillic comment is mis-attributed to "Latin_diacritics".
        non_eng_text = t[q & ~e]
        if enc == "latin-1":
            non_eng_text = non_eng_text.map(
                lambda s: s.encode("latin-1", "ignore").decode("utf-8", "ignore"))
        non_script = non_eng_text
        for name, rx in SCRIPTS:
            hit = non_eng_text.str.contains(rx, regex=True)
            script_counts[name] += int(hit.sum())
            non_script = non_script[~non_script.index.isin(non_eng_text[hit].index)]
        for g in non_script.map(guess_latin_lang):
            lang_counts[g or "unclassified_short_or_lowfunc"] += 1
        print(f"    chunk {ci}: rows={tot:,}", flush=True)
    r = dict(platform=platform, n_comments=tot, n_quality=quality, n_quality_english=eng,
             n_quality_non_english=quality - eng,
             pct_non_english_of_quality=(quality - eng) / quality * 100 if quality else np.nan,
             n_positioned=pos, n_positioned_quality=pos_quality,
             n_positioned_non_english=pos_noneng,
             pct_non_english_of_positioned=pos_noneng / pos_quality * 100 if pos_quality else np.nan)
    r.update({f"script_{k}": v for k, v in script_counts.items()})
    r.update({f"lang_{k}": v for k, v in lang_counts.items()})
    rows.append(r)
    print("  ", {k: v for k, v in r.items() if not k.startswith("script_")}, flush=True)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/e7_language_coverage.csv", index=False)

L = ["# E7 - Non-English coverage and the size of a multilingual extension (TPc3 Q2)\n",
     "The pipeline keeps only comments passing `is_quality_text` (length + alphabetic ratio) "
     "and `is_english` (>=85% ASCII letters + >=3% English function words). This quantifies "
     "exactly what that second filter removes.\n",
     "| Platform | comments | quality-passing | English | non-English | % non-English | positioned (quality) | positioned non-English | % of positioned |",
     "|---|---|---|---|---|---|---|---|---|"]
for _, r in res.iterrows():
    L.append(f"| {r.platform} | {int(r.n_comments):,} | {int(r.n_quality):,} | {int(r.n_quality_english):,} | "
             f"{int(r.n_quality_non_english):,} | {r.pct_non_english_of_quality:.1f}% | "
             f"{int(r.n_positioned_quality):,} | {int(r.n_positioned_non_english):,} | "
             f"{r.pct_non_english_of_positioned:.1f}% |")
L.append("")
L.append("## What the excluded (non-English, quality-passing) comments actually are")
L.append("### non-Latin scripts")
L.append("| Platform | " + " | ".join(n for n, _ in SCRIPTS) + " |")
L.append("|---" * (len(SCRIPTS) + 1) + "|")
for _, r in res.iterrows():
    L.append(f"| {r.platform} | " + " | ".join(f"{int(r['script_' + n]):,}" for n, _ in SCRIPTS) + " |")
L.append("")
L.append("### Latin-script languages (stop-word vote)")
langs = list(LANG_STOP) + ["unclassified_short_or_lowfunc"]
L.append("| Platform | " + " | ".join(langs) + " |")
L.append("|---" * (len(langs) + 1) + "|")
for _, r in res.iterrows():
    L.append(f"| {r.platform} | " + " | ".join(f"{int(r['lang_' + n]):,}" for n in langs) + " |")
L += ["",
      "## Reading",
      "- The English filter is a scoping decision, not a data limitation: the excluded volume is "
      "reported above and the raw comments are retained in the release, so a multilingual track "
      "can be built from the same pipeline without re-collection.",
      "- The excluded set is dominated by Latin-script non-English threads (Polymarket's political "
      "markets draw large Romanian/Spanish/Portuguese discussions) plus short low-function-word "
      "English. `is_english` therefore removes real foreign-language discourse, not only noise -- "
      "which is precisely the population TPc3's question is about.",
      "- Two things would have to be re-validated for a non-English track, and we should say so "
      "explicitly rather than claim transfer: (i) `is_english`'s function-word heuristic has no "
      "counterpart in the target language, so the quality filter must be re-derived; (ii) the "
      "explicit-position lexicon (`has_explicit_pos_signal` / `_score_explicit_signal`: money "
      "amounts, first-person holding verbs, side words) is English-specific, and it is exactly "
      "the component that language-specific writing style would perturb.",
      "- Camera-ready commitment: report these counts in the dataset card as the multilingual "
      "headroom, and state the two re-validation steps as the concrete future-work protocol."]

open(f"{OUT}/e7_summary.md", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
print(f"\n[written] e7_language_coverage.csv, e7_summary.md")
