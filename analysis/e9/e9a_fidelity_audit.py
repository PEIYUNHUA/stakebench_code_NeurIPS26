"""
E9a fidelity audit -- the gate that must pass before any LLM call.

Design decision D5: the biggest risk in the translation manipulation is that translation
destroys the commitment cue itself. If that happens, a drop in the language arm says nothing
about language robustness, it just says the translator lost the signal. Three checks:

  1. Numeral preservation. Amounts, percentages and prices are the one commitment cue that is
     language-independent, so they must survive. Target >= 0.95; below 0.90 the language is
     flagged red and its conclusions are limited (RUNBOOK 3.4 / 6).
  2. Back-translation chrF against the original English.
  3. Cue-family retention: hit rate of each cue family in the original, the translation and
     the back-translation. This is the E9c question -- which cues are quantity-carried and
     survive, and which are English-lexicon-carried and do not.

The 30-per-language human spot check is NOT produced here. This script only exports the
sample to be judged; the human_ok column is left empty on purpose. RUNBOOK 3.4 requires that
column to be truthfully filled by a person, so it is never machine-filled.

Cue definitions are imported from rebuttal/analysis/e9c_cue_portability.py rather than
redefined, so the two analyses cannot drift apart.

Inputs:  e9b_source_instances.csv, e9b_translations*.csv
Outputs: e9a_translation_fidelity.csv    lang, n, numeral_preservation, backtrans_chrf, ...
         e9a_cue_retention.csv           cue family x language x arm hit rates
         e9a_human_spotcheck_TOFILL.csv  30 per language, human_ok column empty
"""
import os, re, sys, json, argparse
import pandas as pd
import numpy as np

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "rebuttal", "analysis"))
OUT = os.path.join(ROOT, "rebuttal", "results", "e9")

from e9c_cue_portability import CUES, QUANTITY          # single source of truth for cues

import sacrebleu

# Numeral preservation is measured on VALUES, not surface strings. Spanish and Romanian swap
# the decimal and thousands separators, so "100,000" -> "100.000" and "3.4" -> "3,4" are
# faithful translations. A string comparison scores those as losses and manufactures a fake
# gate failure -- an early version of this script did exactly that, reporting ron 0.76 /
# spa 0.83 when the quantities were in fact intact. Scale suffixes are also resolved, so
# "100 k" -> "100.000" counts as preserved because it is.
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NUM_TOKEN_RE = re.compile(r"(\d[\d.,]*)\s*([kmb])?\b", re.I)
SCALE = {"k": 1e3, "m": 1e6, "b": 1e9}
SPOTCHECK_N = 30


def _candidates(num, suf):
    """All values a numeric token could denote under either separator convention."""
    vals = set()
    for parsed in (num.replace(",", ""),                       # , = thousands, . = decimal
                   num.replace(".", "").replace(",", "."),     # . = thousands, , = decimal
                   re.sub(r"[.,]", "", num)):                  # digits only
        try:
            v = float(parsed)
        except ValueError:
            continue
        vals.add(v)
        if suf:
            vals.add(v * SCALE[suf.lower()])
    return vals


def numerals(text, primary_only=False):
    """Numeric tokens as value sets. primary_only keeps just the source-side reading."""
    text = URL_RE.sub(" ", str(text))       # URLs carry digits that are not commitment cues
    out = []
    for m in NUM_TOKEN_RE.finditer(text):
        cands = _candidates(m.group(1), m.group(2))
        if not cands:
            continue
        out.append(min(cands) if primary_only else cands)
    return out


def numeral_preservation(src, tgt):
    """Fraction of source quantities still present in the target. None if none present."""
    s = numerals(src)
    if not s:
        return None
    tgt_vals = set()
    for c in numerals(tgt):
        tgt_vals |= c
    return sum(1 for c in s if c & tgt_vals) / len(s)


def cue_rates(texts):
    s = pd.Series([str(t) for t in texts])
    return {k: float(s.str.contains(rx, regex=True).mean()) for k, rx in CUES.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translations", default="e9b_translations.csv")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _smoke")
    args = ap.parse_args()

    src = pd.read_csv(os.path.join(OUT, "e9b_source_instances.csv"))
    tr = pd.read_csv(os.path.join(OUT, args.translations))
    m = tr.merge(src[["comment_id", "task", "text_en", "gt", "question", "platform", "topic"]],
                 on=["comment_id", "task"], how="left")
    m["text_translated"] = m["text_translated"].astype(str)
    m["text_backtranslated"] = m["text_backtranslated"].astype(str)

    fid_rows, cue_rows, spot = [], [], []
    for lang, g in m.groupby("lang"):
        np_fwd = [numeral_preservation(a, b) for a, b in zip(g.text_en, g.text_translated)]
        np_rt = [numeral_preservation(a, b) for a, b in zip(g.text_en, g.text_backtranslated)]
        have = [x for x in np_fwd if x is not None]
        have_rt = [x for x in np_rt if x is not None]

        chrf = sacrebleu.corpus_chrf(g.text_backtranslated.tolist(),
                                     [g.text_en.astype(str).tolist()]).score
        bleu = sacrebleu.corpus_bleu(g.text_backtranslated.tolist(),
                                     [g.text_en.astype(str).tolist()]).score

        # length ratio flags truncation / runaway generation
        lr = (g.text_translated.str.len() / g.text_en.astype(str).str.len().clip(lower=1))

        fid_rows.append(dict(
            lang=lang, n=len(g),
            n_with_numerals=len(have),
            numeral_preservation=round(float(np.mean(have)), 4) if have else None,
            numeral_preservation_backtrans=round(float(np.mean(have_rt)), 4) if have_rt else None,
            backtrans_chrf=round(chrf, 2), backtrans_bleu=round(bleu, 2),
            len_ratio_median=round(float(lr.median()), 3),
            len_ratio_p95=round(float(lr.quantile(0.95)), 3),
            empty_translations=int((g.text_translated.str.strip() == "").sum()),
            human_ok_rate="",           # filled by a person, see e9a_human_spotcheck_TOFILL.csv
            notes="",
        ))

        en_rates = cue_rates(g.text_en)
        xx_rates = cue_rates(g.text_translated)
        rt_rates = cue_rates(g.text_backtranslated)
        for cue in CUES:
            kind = "quantity" if cue in QUANTITY else "lexical"
            cue_rows.append(dict(
                lang=lang, cue=cue, kind=kind,
                rate_en=round(en_rates[cue], 4), rate_translated=round(xx_rates[cue], 4),
                rate_backtranslated=round(rt_rates[cue], 4),
                retention=round(xx_rates[cue] / en_rates[cue], 4) if en_rates[cue] > 0 else None,
                caveat=("detector is an English regex; a low rate on translated text means the "
                        "English cue detector does not port, NOT that the cue was destroyed"
                        if kind == "lexical" else
                        "partly portable: digits survive, the currency/unit word may not"),
            ))

        s = g.sample(min(SPOTCHECK_N, len(g)), random_state=42)
        for _, r in s.iterrows():
            spot.append(dict(lang=lang, comment_id=r.comment_id, task=r.task,
                             question=str(r.question)[:200], text_en=r.text_en,
                             text_translated=r.text_translated,
                             text_backtranslated=r.text_backtranslated, gt=r["gt"],
                             human_ok="", reviewer="", comment=""))

    fid = pd.DataFrame(fid_rows)
    cue = pd.DataFrame(cue_rows)
    tag = args.tag
    fid.to_csv(os.path.join(OUT, f"e9a_translation_fidelity{tag}.csv"), index=False, encoding="utf-8")
    cue.to_csv(os.path.join(OUT, f"e9a_cue_retention{tag}.csv"), index=False, encoding="utf-8")
    pd.DataFrame(spot).to_csv(os.path.join(OUT, f"e9a_human_spotcheck_TOFILL{tag}.csv"),
                              index=False, encoding="utf-8")

    print(fid.drop(columns=["human_ok_rate", "notes"]).to_string(index=False))
    print("\n--- cue retention (translated / original) ---")
    piv = cue.pivot(index="cue", columns="lang", values="retention")
    piv.insert(0, "kind", [("quantity" if c in QUANTITY else "lexical") for c in piv.index])
    print(piv.to_string())
    print("\n  READ THIS BEFORE QUOTING THE LEXICAL ROWS: every cue detector is an English\n"
          "  regex. On translated text a lexical retention of 0.00 means the English detector\n"
          "  stopped firing, not that the commitment cue was destroyed -- a Spanish comment can\n"
          "  state a position perfectly well without matching an English trade-verb pattern.\n"
          "  The 'side word' row shows the reverse failure: English \\b(yes|no)\\b over-fires on\n"
          "  Spanish 'no', giving retention > 1. These rows measure detector portability, which\n"
          "  is the component TPc3's second half asks about; they are not a cue-loss measurement.")

    # gate
    bad = fid[(fid.numeral_preservation.notna()) & (fid.numeral_preservation < 0.90)]
    print("\n=== GATE ===")
    if len(bad):
        print("STOP: numeral preservation below 0.90 for: " + ", ".join(bad.lang))
        print("RUNBOOK 10.1 -- report back before scaling these languages.")
    else:
        print("numeral preservation >= 0.90 for all languages")
    print("human spot check: NOT machine-filled. Fill e9a_human_spotcheck_TOFILL.csv "
          "(human_ok column) before quoting a human_ok_rate.")


if __name__ == "__main__":
    main()
