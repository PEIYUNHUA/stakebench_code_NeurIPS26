"""
E9a: translate the frozen English instances into the Tier-1 languages and back.

Design decision D3: the translator must be decoupled from the evaluated models, so this uses
NLLB-200 (facebook/nllb-200-distilled-1.3B) locally. Using an evaluated LLM to translate and
then evaluating with the same family would confound the language effect with that family's
own expression preferences.

Design decision D4: every language also gets a back-translation arm (en -> xx -> en). Without
it, any "low-resource languages score worse" result can be dismissed with "your translation
was bad in the first place".

Only the comment text is translated. Market questions, prompts, labels and positions are
untouched -- that is the definition of the controlled manipulation.

Usage:
  python rebuttal/analysis/e9a_translate.py              # all frozen instances
  python rebuttal/analysis/e9a_translate.py --limit 30   # smoke subset

Inputs:  rebuttal/results/e9/e9b_source_instances.csv
Outputs: rebuttal/results/e9/e9b_translations.csv   comment_id, lang, text_translated,
                                                    text_backtranslated
         rebuttal/results/e9/e9a_translation_params.json
"""
import os, sys, json, time, argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "rebuttal", "results", "e9")

MODEL_ID = "facebook/nllb-200-distilled-1.3B"
SRC = "eng_Latn"
LANGS = ["spa_Latn", "ron_Latn", "swh_Latn"]      # Tier 1, frozen pre-registration
GEN = dict(num_beams=4, max_length=256, no_repeat_ngram_size=0, do_sample=False)

# GEN is pre-registered and must not change. BATCH is not a scientific parameter -- padding is
# masked, so it does not alter the beams -- and every batch is appended to a checkpoint so a
# crash costs one batch, not the whole run.
#
# Peak VRAM is set by beam search, not by the weights. _reorder_cache index_selects the whole
# past_key_values every step, so the transient peak is weights + 2x KV cache:
#   2.6 GB + 2 * (BATCH * 4 beams * 256 tokens * 24 layers * 2 * 1024 * 2 bytes)
# At BATCH=32 that is ~10 GB and an 8 GB card dies -- which is what happened on the ron
# back-translation pass, on a degenerate long generation (GEN has no_repeat_ngram_size=0, so
# a repeat loop can run to the full 256 tokens). BATCH=12 puts the peak near 4.5 GB.
BATCH = 12
EMPTY_CACHE_EVERY = 8       # allocator fragmentation, not total size, killed the last batch
CKPT_DIR = "e9a_translate_ckpt"


def is_oom(exc):
    """True for both allocator OOM and the raw CUDA OOM that beam reorder raises.

    torch.cuda.OutOfMemoryError only covers the caching allocator. A failure inside
    _reorder_cache surfaces as a plain RuntimeError('CUDA error: out of memory'), which an
    except-OutOfMemoryError clause silently misses -- that is how the first full run lost its
    process. That variant also poisons the CUDA context, so the in-process retry below cannot
    always recover; restarting the process resumes from the checkpoints.
    """
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


def lang_id(tok, code):
    """forced_bos_token_id across transformers versions (lang_code_to_id was removed)."""
    lid = tok.convert_tokens_to_ids(code)
    unk = getattr(tok, "unk_token_id", None)
    if lid is None or lid == unk:
        mapping = getattr(tok, "lang_code_to_id", None)
        if mapping and code in mapping:
            return mapping[code]
        raise SystemExit(f"cannot resolve NLLB language id for {code}")
    return lid


def translate(tok, model, texts, src, tgt, device, tag="", ckpt=None):
    """Translate texts, appending each batch to `ckpt` so the pass is resumable.

    Output order is the input order, so a checkpoint of length n means the first n texts are
    done and the pass restarts at texts[n]. On CUDA OOM the batch is halved and retried; the
    reduced size persists for the rest of the pass.
    """
    tok.src_lang = src
    bos = lang_id(tok, tgt)
    out, t0 = [], time.time()

    if ckpt and os.path.exists(ckpt):
        out = pd.read_csv(ckpt, keep_default_na=False)["text_out"].astype(str).tolist()[:len(texts)]
        if out:
            print(f"  [{tag}] resuming at {len(out)}/{len(texts)}", flush=True)
    if len(out) >= len(texts):
        return out[:len(texts)]

    batch, nb = BATCH, 0
    while len(out) < len(texts):
        i = len(out)
        chunk = [t if t.strip() else "." for t in texts[i:i + batch]]
        try:
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=bos, **GEN)
            dec = tok.batch_decode(gen, skip_special_tokens=True)
        except Exception as exc:
            if not is_oom(exc):
                raise
            enc = gen = None
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            print(f"  [{tag}] CUDA OOM at batch {batch * 2} -> retrying at {batch}", flush=True)
            continue
        enc = gen = None
        nb += 1
        if device == "cuda" and nb % EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()
        out.extend(dec)
        if ckpt:
            pd.DataFrame(dict(idx=range(i, i + len(dec)), text_out=dec)).to_csv(
                ckpt, mode="a", header=not os.path.exists(ckpt), index=False, encoding="utf-8")
        print(f"  [{tag}] {len(out)}/{len(texts)}  {time.time()-t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="translate only the first N per task")
    ap.add_argument("--out", default="e9b_translations.csv")
    args = ap.parse_args()

    src = pd.read_csv(os.path.join(OUT, "e9b_source_instances.csv"))
    if args.limit:
        src = src.groupby("task", group_keys=False).head(args.limit).reset_index(drop=True)
    texts = src["text_en"].astype(str).tolist()
    print(f"instances: {len(src)}  (G1={sum(src.task=='G1')} G2={sum(src.task=='G2')})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_ID} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    model.eval()

    ckpt_dir = os.path.join(OUT, CKPT_DIR + ("_" + args.out.replace(".csv", "")))
    os.makedirs(ckpt_dir, exist_ok=True)

    rows = []
    for lang in LANGS:
        print(f"\n=== {SRC} -> {lang} ===", flush=True)
        fwd = translate(tok, model, texts, SRC, lang, device, tag=lang,
                        ckpt=os.path.join(ckpt_dir, f"{lang}_fwd.csv"))
        print(f"=== {lang} -> {SRC} (back-translation) ===", flush=True)
        back = translate(tok, model, fwd, lang, SRC, device, tag=lang + "_rt",
                         ckpt=os.path.join(ckpt_dir, f"{lang}_back.csv"))
        for cid, task, f, b in zip(src.comment_id, src.task, fwd, back):
            rows.append(dict(comment_id=cid, task=task, lang=lang,
                             text_translated=f, text_backtranslated=b))
        if device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, args.out)
    df.to_csv(path, index=False, encoding="utf-8")

    with open(os.path.join(OUT, "e9a_translation_params.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(model=MODEL_ID, src=SRC, languages=LANGS, generation=GEN,
                       batch_size=BATCH,
                       batch_size_note="initial batch; halved and retried on CUDA OOM. "
                                       "Padding is masked, so batching does not affect output.",
                       device=device, dtype=str(model.dtype),
                       transformers=__import__("transformers").__version__,
                       n_instances=int(len(src)), limit=args.limit), fh, indent=2)

    print(f"\nwrote {path}  rows={len(df)}")


if __name__ == "__main__":
    main()
