"""
V16 shared benchmark module.

This file contains the reusable core logic for CommitBench V18:
- statistics helpers
- backend and prompt helpers
- data loading and normalization
- task construction for T0/T1/T2/T3
- phase-0 descriptive analysis
- reporting utilities

V16 changes (over V15):
- parsing.py: fix <think>…</think> full block stripping (was tag-only in V15)
- llm_adapter.py: add n_params log + 70B safety check; add OpenRouter backend
- model_profiles.py: fix fallback alias bug; fix llama3:70b → Instruct; add OpenRouter profiles
- All [V15] log prefixes updated to [V18]

Pipeline:
T0 Position Detection -> T1 Belief Inference -> T2 Language-Action Consistency -> T3 Belief-Price Lead
"""


import os
import json
import logging
import random
import threading
import time
import re
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from .config import *
except ImportError:
    from config import *



def binom_pvalue_greater(k: int, n: int, p: float = 0.5) -> float:
    """Return a one-sided binomial tail probability P(X >= k | n, p)."""
    if n <= 1000:
        from math import comb
        return sum(comb(n, i) * (p**i) * ((1-p)**(n-i)) for i in range(k, n+1))
    mu    = n * p
    sigma = math.sqrt(n * p * (1 - p))
    z     = (k - 0.5 - mu) / sigma
    return _norm_sf(z)


def mannwhitney_pvalue_greater(x: list, y: list) -> float:
    """Approximate a one-sided Mann-Whitney U p-value for x > y."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 1.0
    u_x = (sum(1 for xi in x for yi in y if xi > yi)
           + 0.5 * sum(1 for xi in x for yi in y if xi == yi))
    mu    = nx * ny / 2
    sigma = math.sqrt(nx * ny * (nx + ny + 1) / 12)
    return 1.0 if sigma == 0 else _norm_sf((u_x - mu) / sigma)


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def spearman_rho(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Return Spearman rho and an approximate two-sided p-value."""
    n = len(x)
    if n < 4:
        return 0.0, 1.0

    def _rank(lst):
        order = sorted(range(n), key=lambda i: lst[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and lst[order[j]] == lst[order[j+1]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k2 in range(i, j+1):
                ranks[order[k2]] = avg
            i = j + 1
        return ranks

    rx, ry = _rank(list(x)), _rank(list(y))
    mx, my = sum(rx)/n, sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    dx  = math.sqrt(sum((rx[i]-mx)**2 for i in range(n)))
    dy  = math.sqrt(sum((ry[i]-my)**2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0, 1.0
    rho = max(-1.0, min(1.0, num / (dx * dy)))
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * math.sqrt((n-2) / (1 - rho**2))
    return rho, 2 * _norm_sf(abs(t))


# Pipeline stage metadata used by summary reports.
PIPELINE_STAGES = [
    (0, "T0", "Position Detection",    "binary",    0.5,  "acc"),
    (1, "T1", "Belief Inference",      "2c/DA",     0.5,  "t1_acc_2c"),
    (2, "T2", "Language-Action",       "4c",        0.25, "t2_acc_4c"),
    (3, "T3", "Belief-Price Lead",     "stat_3c",   0.5,  "t3_stat_3c"),
]
# (task_idx, short_name, full_name, metric_mode, random_baseline, summary_key)

# Regex for explicit first-person position ownership language used by Task0.
_EXPLICIT_POS_SIGNAL_RE = re.compile(
    r'\$\s*\d|\d[\d,]*\s*(?:dollar|buck)s?\b'                        # Dollar amount
    r'|\bmy\s+(?:position|bet|bets?|shares?|stake|hold(?:ing)?)\b'   # Ownership phrase
    r'|\bi(?:\s+have)?\s+(?:bought|sold|entered|hold(?:ing)?|added|cut|exited|went)\b'  # Action verb
    r'|\b(?:long|short|holding|in)\s+(?:yes|no)\b'                   # Explicit side
    r'|\btook\s+(?:profit|a\s+position)\b'                           # Position management
    r'|\bmy\s+\$',                                                   # "my $200"
    re.IGNORECASE
)


random.seed(SEED)
np.random.seed(SEED)


def _sample_df(df: pd.DataFrame, n: Optional[int] = None) -> pd.DataFrame:
    if RUN_FULL or n is None or n >= len(df):
        return df.copy()
    return df.sample(n, random_state=SEED).copy()


# Logging helpers

logger = logging.getLogger("commitbench")


def _setup_logging(log_dir: str) -> str:
    """Initialize console and file logging, then return the log file path."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # File logger
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    logger.info(f"Log file: {log_path}")
    return log_path


def _log(msg: str, level: str = "info"):
    """Write a message to both stdout and the configured logger."""
    print(msg)
    getattr(logger, level)(msg)


# Resume helpers

# Completion threshold for considering a CSV output effectively done.
_COMPLETE_RATIO = 0.90

# Runtime path for resumable state; assigned by the entrypoint.
_RUN_STATE_PATH = ""

def _state_load() -> dict:
    if _RUN_STATE_PATH and os.path.exists(_RUN_STATE_PATH):
        try:
            with open(_RUN_STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _state_mark_done(key: str):
    """Mark a task key as completed in the resumable run-state file."""
    if not _RUN_STATE_PATH:
        return
    state = _state_load()
    state[key] = {"done": True, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(_RUN_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _state_is_done(key: str) -> bool:
    return _state_load().get(key, {}).get("done", False)


# Columns that are always boolean by design; normalised on CSV load even if a row
# has a stray non-bool value (e.g. "YES" from a shifted CSV column).
_KNOWN_BOOL_COLS = {
    "correct", "has_position",
    "llm_correct", "naive_correct",
    "stat_correct_3c", "stat_correct_2c",
}


def _norm_bool(series: "pd.Series") -> "pd.Series":
    """Convert a column to bool robustly: 'True'→True, anything else→False."""
    return series.astype(str).str.strip().str.lower() == "true"


def _fix_bool_cols(df: "pd.DataFrame") -> "pd.DataFrame":
    """Return a copy of df with all known boolean columns normalised to bool dtype."""
    needs_fix = [c for c in _KNOWN_BOOL_COLS if c in df.columns and df[c].dtype == object]
    if not needs_fix:
        return df
    df = df.copy()
    for c in needs_fix:
        df[c] = _norm_bool(df[c])
    return df


def _csv_load_partial(csv_path: str) -> pd.DataFrame:
    """Safely read a partial CSV checkpoint and return an empty frame on failure."""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return pd.DataFrame()
        for col in df.columns:
            if col not in _KNOWN_BOOL_COLS:
                continue
            if df[col].dtype != object:
                continue
            df[col] = _norm_bool(df[col])
        return df
    except Exception:
        return pd.DataFrame()


def _is_complete(csv_path: str, expected_rows: int) -> bool:
    """Return True when a checkpoint CSV is complete enough to skip rerunning."""
    df = _csv_load_partial(csv_path)
    return len(df) >= max(1, int(expected_rows * _COMPLETE_RATIO))


def _run_infer_loop(
    sample_rows: list,
    done_keys: set,
    worker_fn,
    existing_results: list,
    csv_path: str,
    expected: int,
    checkpoint_every: int = 20,
) -> list:
    """Run LLM inference, dispatching to the appropriate backend strategy.

    Parameters
    ----------
    sample_rows      : list of (row_key_tuple, row_item) pairs.
    done_keys        : set of already-processed row_key_tuples (resume support).
    worker_fn        : callable(row_item) -> dict | None.  None = skip sample.
    existing_results : pre-existing result dicts loaded from the partial CSV.
    csv_path         : checkpoint CSV path.
    expected         : expected total row count (for checkpoint log messages).
    checkpoint_every : flush CSV every N newly completed results.
    """
    pending = [(k, item) for k, item in sample_rows if k not in done_keys]
    results = list(existing_results)

    if not pending:
        return results

    # Batch backends: collect all prompts first, batch-generate, then replay.
    # vLLM uses PagedAttention; transformers uses native left-padded generate().
    if BACKEND in ("vllm", "transformers"):
        return _vllm_batch_infer_loop(
            pending, worker_fn, results, csv_path, expected, checkpoint_every
        )

    concurrency = OPENROUTER_CONCURRENCY if BACKEND == "openrouter" else 1

    if concurrency <= 1:
        for _, item in pending:
            r = worker_fn(item)
            if r is not None:
                results.append(r)
                if len(results) % checkpoint_every == 0:
                    pd.DataFrame(results).to_csv(csv_path, index=False)
                    logger.debug(f"checkpoint: {len(results)}/{expected}")
        return results

    _lock = threading.Lock()
    _log(f"  [PARALLEL] workers={concurrency}  pending={len(pending)}")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(worker_fn, item): key for key, item in pending}
        for future in as_completed(futures):
            try:
                r = future.result()
            except Exception as exc:
                _log(f"  [PARALLEL][ERR] {type(exc).__name__}: {exc}", "error")
                continue
            if r is not None:
                with _lock:
                    results.append(r)
                    n = len(results)
                    if n % checkpoint_every == 0:
                        pd.DataFrame(results).to_csv(csv_path, index=False)
                        logger.debug(f"checkpoint: {n}/{expected}")

    return results


def _vllm_batch_infer_loop(
    pending: list,
    worker_fn,
    existing_results: list,
    csv_path: str,
    expected: int,
    checkpoint_every: int,
) -> list:
    """Two-pass batch inference for vLLM.

    Pass 1 — Capture: temporarily replaces query_llm with a prompt-capture
    stub, runs every worker to collect prompts (results are discarded).

    Pass 2 — Batch generate: submits all captured prompts to vLLM in one call
    via the module-level _vllm_query_batch (wired to LLMAdapter.query_batch
    by runtime.py).

    Pass 3 — Replay: temporarily replaces query_llm with a cached-response
    stub, re-runs workers to produce final result dicts (which are saved).
    """
    import sys
    mod = sys.modules[__name__]

    original_query = mod.query_llm
    results = list(existing_results)

    # ── Pass 1: capture prompts ──────────────────────────────────────────────
    captured: list[str] = []

    def _capture(prompt, model=None):
        captured.append(prompt)
        return "__VLLM_PENDING__"

    mod.query_llm = _capture
    try:
        for _, item in pending:
            worker_fn(item)
    finally:
        mod.query_llm = original_query

    if not captured:
        _log("[vLLM BATCH] no prompts captured — nothing to generate")
        return results

    _log(f"[vLLM BATCH] captured {len(captured)} prompts — starting batch generation")

    # ── Pass 2: batch generate ───────────────────────────────────────────────
    normalized_responses = _vllm_query_batch(captured)

    _log(f"[vLLM BATCH] batch generation complete ({len(normalized_responses)} responses)")

    # ── Pass 3: replay with cached responses ────────────────────────────────
    _replay_idx = [0]

    def _replay(prompt, model=None):
        idx = _replay_idx[0]
        _replay_idx[0] = idx + 1
        return normalized_responses[idx]

    mod.query_llm = _replay
    try:
        for _, item in pending:
            r = worker_fn(item)
            if r is not None:
                results.append(r)
                if len(results) % checkpoint_every == 0:
                    pd.DataFrame(results).to_csv(csv_path, index=False)
                    logger.debug(f"checkpoint: {len(results)}/{expected}")
    finally:
        mod.query_llm = original_query

    return results


# Ollama backend

def query_ollama(prompt: str, model: str = None) -> str:
    """Send a prompt to the Ollama backend and return the raw text response."""
    _model = model or MODEL
    payload = {
        "model": _model, "prompt": prompt + "\n/no_think",
        "temperature": TEMPERATURE, "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()["response"].strip()
        return raw.split("</think>")[-1].strip() if "<think>" in raw else raw
    except Exception as e:
        return f"ERROR: {e}"


# Transformers backend

def _load_hf_model(model_name: str):
    """Lazily load and cache the requested HuggingFace model."""
    global _hf_model, _hf_tokenizer, _hf_loaded_name
    hf_name = HF_MODEL_MAP.get(model_name, model_name)
    if _hf_loaded_name == hf_name:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _log(f"[Transformers] Loading model: {hf_name} ...")
    _hf_tokenizer = AutoTokenizer.from_pretrained(
        hf_name, trust_remote_code=True, cache_dir=HF_CACHE_DIR)
    _hf_model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True, cache_dir=HF_CACHE_DIR
    )
    _hf_loaded_name = hf_name
    _log(f"[Transformers] Model ready: {hf_name}")


def query_transformers(prompt: str, model: str = None) -> str:
    """Run local generation with HuggingFace transformers."""
    import torch
    _model = model or MODEL
    _load_hf_model(_model)
    prompt_text = prompt + "\n/no_think"
    # Chat models use their chat template; base models use the raw prompt.
    if getattr(_hf_tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt_text}]
        try:
            text = _hf_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = _hf_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt_text
    inputs = _hf_tokenizer(text, return_tensors="pt").to(_hf_model.device)
    with torch.no_grad():
        gen_kwargs = dict(
            max_new_tokens=256,
            pad_token_id=_hf_tokenizer.eos_token_id,
        )
        if TEMPERATURE > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = TEMPERATURE
        else:
            gen_kwargs["do_sample"] = False
        outputs = _hf_model.generate(**inputs, **gen_kwargs)
    raw = _hf_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return raw.split("</think>")[-1].strip() if "<think>" in raw else raw


# Unified LLM dispatch

def query_llm(prompt: str, model: str = None) -> str:
    """Dispatch generation to the configured backend."""
    if BACKEND == "transformers":
        return query_transformers(prompt, model)
    return query_ollama(prompt, model)


def _vllm_query_batch(prompts: list, model=None) -> list:
    """Batch query (vLLM or transformers batch path).

    Replaced at runtime by LLMAdapter.query_batch when BACKEND='vllm' or
    BACKEND='transformers'. Sequential fallback used for other backends.
    """
    return [query_llm(p, model) for p in prompts]


def check_backend():
    """Check whether the configured LLM backend is available."""
    if BACKEND == "transformers":
        try:
            import transformers, torch
            _log(f"[OK] Transformers backend ready (transformers={transformers.__version__}, "
                 f"torch={torch.__version__}, CUDA={torch.cuda.is_available()})")
            for m in MODEL_LIST:
                hf_name = HF_MODEL_MAP.get(m, m)
                _log(f"  Model mapping:: {m} ->{hf_name}")
            return True
        except ImportError as e:
            _log(f"[ERR] Transformers backend unavailable: {e}", "error")
            return False
    else:
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            available = [m["name"] for m in r.json().get("models", [])]
            _log(f"[OK] Ollama connection OK, available models: {available}")
            for m in MODEL_LIST:
                if not any(m.split(":")[0] in a for a in available):
                    _log(f"[!!] Model {m} not found, please first run: ollama pull {m}", "error")
            return True
        except Exception as e:
            _log(f"[ERR] Ollama unavailable: {e}", "error")
            return False


#  Prompt design

def prompt_task0(question: str, text: str) -> str:
    """Build the one-shot prompt for Task0 position detection."""
    feats = _task0_text_features(text)
    feat_line = (
        f"[Pre-extracted text features: "
        f"dollar_amount={feats['dollar_amount']}, "
        f"ownership_phrase={feats['ownership_phrase']}, "
        f"action_verb={feats['action_verb']}, "
        f"first_person={feats['first_person']}]"
    )
    return f"""You are a classifier. Decide whether the author of a prediction market comment CURRENTLY HOLDS a financial position (bet) in the market.

Market question: "{question}"
Comment: "{text}"
{feat_line}

Look for these STRUCTURAL SIGNALS that indicate a held position:
  1. Dollar amounts or stake sizes ("I put in $50", "my $200 position", "bet $100")
  2. Explicit ownership phrases ("my position", "my bet", "my shares", "I hold")
  3. First-person buy/sell actions ("I bought YES", "I sold NO", "I went long", "I entered")
  4. Explicit YES/NO holding statement ("I'm in YES", "holding NO", "long YES")
  5. Position management language ("adding to my position", "took profits", "cutting my NO")

Use the pre-extracted features as a guide, but base your final decision on whether the COMMENT TEXT clearly expresses first-person ownership (not just financial discussion by an observer).

If ANY of signals 1-5 are present in the comment, answer YES.
If NONE are present, answer NO.

Example:
  Comment: "I put $50 on YES because the poll numbers look strong"
  Answer: YES

Reply with exactly ONE word: YES or NO
Answer:""".strip()


def prompt_task0_zeroshot(question: str, text: str) -> str:
    """Zero-shot version of Task0, used to compare against the one-shot prompt."""
    feats = _task0_text_features(text)
    feat_line = (
        f"[Pre-extracted text features: "
        f"dollar_amount={feats['dollar_amount']}, "
        f"ownership_phrase={feats['ownership_phrase']}, "
        f"action_verb={feats['action_verb']}, "
        f"first_person={feats['first_person']}]"
    )
    return f"""You are a classifier. Decide whether the author of a prediction market comment CURRENTLY HOLDS a financial position (bet) in the market.

Market question: "{question}"
Comment: "{text}"
{feat_line}

Look for these STRUCTURAL SIGNALS that indicate a held position:
  1. Dollar amounts or stake sizes ("I put in $50", "my $200 position", "bet $100")
  2. Explicit ownership phrases ("my position", "my bet", "my shares", "I hold")
  3. First-person buy/sell actions ("I bought YES", "I sold NO", "I went long", "I entered")
  4. Explicit YES/NO holding statement ("I'm in YES", "holding NO", "long YES")
  5. Position management language ("adding to my position", "took profits", "cutting my NO")

Use the pre-extracted features as a guide, but base your final decision on whether the COMMENT TEXT clearly expresses first-person ownership (not just financial discussion by an observer).

If ANY of signals 1-5 are present in the comment, answer YES.
If NONE are present, answer NO.

Reply with exactly ONE word: YES or NO
Answer:""".strip()


def prompt_task1(question: str, text: str) -> str:
    """Build the prompt for Task1 belief-direction inference."""
    return f"""You are an expert at analyzing prediction market comments.

Market question: "{question}"
Comment: "{text}"

Infer which direction the commenter believes this market will resolve.

Labels:
  YES      = The commenter believes, expects, hopes, or implies the market resolves YES
  NO       = The commenter believes, expects, doubts, or implies the market resolves NO
  NEUTRAL  = Use ONLY when the comment is purely factual/procedural with absolutely zero directional implication

IMPORTANT rules:
  - Even a weak, indirect, or hedged signal counts ->choose YES or NO
  - Skepticism, doubt, or worry about the YES outcome ->NO
  - Optimism, support, or expectation of YES ->YES
  - Do NOT output NEUTRAL just because the commenter sounds uncertain — uncertainty about magnitude still implies a direction
  - NEUTRAL is reserved for comments like "when does this market close?" or "interesting question" with zero stance

Examples:
  Q: "Will X win the election?"  Comment: "X has been polling ahead lately"  ->YES
  Q: "Will X win the election?"  Comment: "I'm not sure X can pull this off"  ->NO
  Q: "Will X win the election?"  Comment: "What are the resolution criteria?"  ->NEUTRAL

Reply with exactly ONE word: YES / NO / NEUTRAL
Answer:""".strip()


def prompt_task2(question: str, text: str, position_side: str, position_size: float) -> str:
    """Build the 4-class action prompt for Task2."""
    opposite = "NO" if position_side == "YES" else "YES"
    return f"""You are analyzing a prediction market comment alongside the commenter's current position.

Market question: "{question}"
Comment: "{text}"
Commenter's current position: {position_side} (${position_size:,.0f} at stake)

Predict what the commenter will MOST LIKELY do with their position in the NEXT 48 HOURS.

Language signals for each action:
  flip     ->Comment expresses the OPPOSITE belief from their current position ({opposite}).
             Signals: changed mind, new evidence contradicts position, "I was wrong", "switching to {opposite}"
  increase ->Comment shows STRONGER conviction in their current position ({position_side}).
             Signals: adding more, doubling down, very confident, "loading up", bullish/bearish escalation
  decrease ->Comment shows DOUBT, WORRY, or HEDGING about their current position.
             Signals: not sure anymore, risk getting too high, taking some profit, "trimming", cautious tone
  hold     ->Comment is neutral observation or shows no change in conviction.
             Signals: just watching, no strong new opinion, reiterating existing view without escalation

IMPORTANT: Do NOT default to hold. If the comment shows any doubt (->decrease) or opposite view (->flip), choose that.

Example:
  Position: YES ($500). Comment: "The latest data makes me nervous, might be overexposed here." ->decrease
  Position: NO ($200).  Comment: "Everything points to this resolving NO, very confident." ->increase
  Position: YES ($100). Comment: "Actually I think this resolves NO now." ->flip

Reply with exactly ONE word: flip / increase / decrease / hold
Answer:""".strip()


def prompt_task2_flip_hold(question: str, text: str, position_side: str, position_size: float) -> str:
    """Build the binary flip-vs-hold prompt for Task2."""
    opposite = "NO" if position_side == "YES" else "YES"
    return f"""You are analyzing a prediction market comment alongside the commenter's current position.

Market question: "{question}"
Comment: "{text}"
Commenter's current position: {position_side} (${position_size:,.0f} at stake)

Decide whether the commenter's comment suggests they will FLIP their position (switch from {position_side} to {opposite}),
or HOLD (keep their position, regardless of conviction level).

flip ->The comment clearly expresses a belief in the OPPOSITE outcome ({opposite}).
       Signals: "I think this resolves {opposite}", "switched my mind", contradicts their own {position_side} position.
hold ->The comment is neutral, re-affirms their position, or only shows mild doubt (not enough to flip).

Reply with exactly ONE word: flip / hold
Answer:""".strip()


def prompt_task2_decrease_hold(question: str, text: str, position_side: str, position_size: float) -> str:
    """Build the binary decrease-vs-hold prompt for Task2."""
    return f"""You are analyzing a prediction market comment alongside the commenter's current position.

Market question: "{question}"
Comment: "{text}"
Commenter's current position: {position_side} (${position_size:,.0f} at stake)

Decide whether the commenter's comment suggests they will DECREASE their position (reduce exposure, take partial profit, hedge),
or HOLD (keep the same position size).

decrease ->The comment shows DOUBT, WORRY, CAUTION, or PROFIT-TAKING about their {position_side} position.
           Signals: "not as confident", "taking some profit", "trimming", "too risky", hedging language.
hold     ->The comment is neutral, observational, or re-affirms conviction without expressing any doubt.

Reply with exactly ONE word: decrease / hold
Answer:""".strip()


def prompt_task3_lead(question: str, comments_text: str, current_odds: float) -> str:
    lean = "leaning YES (bullish)" if current_odds > 0.5 else "leaning NO (bearish)"
    return f"""You are analyzing a prediction market time window.

Market question: "{question}"
Current market probability (YES): {current_odds:.2f} [{lean}]

Recent comments from the past 7 days:
{comments_text}

Based on the OVERALL DIRECTION implied by these comments, predict whether the market's YES probability will INCREASE or DECREASE over the NEXT 7 days.

Reply with exactly ONE word: UP (probability will increase) / DOWN (probability will decrease) / NEUTRAL
Answer:""".strip()


#  Text utilities

_URL_RE = re.compile(r"https?://\S+")

_STOPWORDS = frozenset({
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","need",
    "this","that","these","those","it","its","i","you","he","she","we","they",
    "me","him","her","us","them","my","your","his","our","their","what","which",
    "who","how","when","where","why","if","not","no","so","as","by","from","up",
    "about","than","then","just","also","very","more","some","any","all","both",
    "each","few","most","other","such","own","same","too","s","t","m","re","ve",
    "ll","d","http","https","www","rt","via","amp","gt","lt","co","like","get",
    "got","one","two","go","going","make","made","know","said","say","think",
    "come","see","look","use","good","time","way","back","give","take","want",
    "here","there","now","new","well","even","still","already","much","many",
    "people","market","markets","bet","betting","odds","prediction","resolve",
    "resolution","yes","no","yeah","nope","yep","ok","okay","lol","lmao","haha",
})

_CERTAINTY_WORDS = frozenset({
    "definitely","certainly","clearly","obviously","absolutely","surely",
    "undoubtedly","indeed","guaranteed","certain","sure","confident","must",
    "always","proven","confirmed","obvious","guarantee","without","clearly",
})
_UNCERTAINTY_WORDS = frozenset({
    "maybe","perhaps","possibly","uncertain","unsure","probably","unlikely",
    "guess","wonder","suppose","seem","appear","unclear","doubt","depends",
    "potentially","conceivably","hopefully","might","perhaps","unsure",
})

_EN_FUNCTION = frozenset({
    "the","a","an","is","are","was","were","i","you","he","she","it","we",
    "they","this","that","in","on","at","to","for","of","and","or","but",
    "not","with","have","has","had","will","would","can","could","do",
})

# Blacklist for clearly non-financial markets such as word games and puzzles.
_NONFINANCIAL_BLACKLIST = frozenset({
    "wordle", "5-letter", "nyt spelling", "spelling bee",
    "crossword", "daily word", "mini crossword", "letterle",
    "quordle", "heardle", "octordle", "worldle", "nerdle",
    "connections puzzle", "strands puzzle",
})


def is_quality_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    clean = _URL_RE.sub("", text).strip()
    if len(clean) < MIN_TEXT_LEN:
        return False
    return sum(c.isalpha() for c in clean) / max(len(clean), 1) >= 0.3


def is_english(text: str) -> bool:
    """Heuristic English detector based on ASCII ratio and function-word coverage."""
    if not isinstance(text, str):
        return False
    clean   = _URL_RE.sub("", text).lower()
    letters = [c for c in clean if c.isalpha()]
    if not letters:
        return False
    if sum(ord(c) < 128 for c in letters) / len(letters) < 0.85:
        return False
    words = clean.split()
    return bool(words) and sum(w in _EN_FUNCTION for w in words) / len(words) >= 0.03


def has_explicit_pos_signal(text: str) -> bool:
    """Return True when text contains explicit first-person holding language."""
    return bool(_EXPLICIT_POS_SIGNAL_RE.search(str(text)))


def _score_explicit_signal(text: str) -> int:
    """Score explicit-position strength on four independent lexical dimensions."""
    t = str(text)
    score = 0
    # Dimension 1: money amount
    if re.search(r'\$\s*\d|\d[\d,]*\s*(?:dollar|buck)s?\b', t, re.I):
        score += 1
    # Dimension 2: action verb
    if re.search(r'\bi(?:\s+have)?\s+(?:bought|sold|entered|hold(?:ing)?|added|cut|exited|went)\b'
                 r'|\btook\s+(?:profit|a\s+position)\b', t, re.I):
        score += 1
    # Dimension 3: explicit YES/NO side
    if re.search(r'\b(?:long|short|holding|in|bought|sold)\s+(?:yes|no)\b'
                 r'|\bmy\s+(?:yes|no)\s+(?:position|bet|shares?)\b', t, re.I):
        score += 1
    # Dimension 4: sports-style betting declaration
    if re.search(r'\bi(?:\s+(?:have|had))?\s+(?:bet|placed\s+a\s+bet|wagered|backed)\b'
                 r"|\bi'?m\s+(?:backing|going\s+with|rooting\s+for)\b"
                 r'|\bmy\s+(?:money|wager|pick)\s+(?:is|on)\b'
                 r'|\bpicked\s+(?:the|them|him|her|it)\b', t, re.I):
        score += 1
    return score


def _task0_text_features(text: str) -> dict:
    """Extract shallow lexical features used to stabilize the Task0 prompt."""
    t = str(text)
    return {
        "dollar_amount":    bool(re.search(r'\$\s*\d|\d[\d,]*\s*(?:dollar|buck)s?\b', t, re.I)),
        "ownership_phrase": bool(re.search(r'\bmy\s+(?:position|bet|shares?|stake|hold(?:ing)?)\b', t, re.I)),
        "action_verb":      bool(re.search(r'\bi(?:\s+have)?\s+(?:bought|sold|entered|hold(?:ing)?|added|cut|exited)\b', t, re.I)),
        "first_person":     bool(re.search(r'\b(?:I|my|me|mine)\b', t)),
    }


def _tokenize(text: str) -> List[str]:
    t = _URL_RE.sub("", str(text).lower())
    t = re.sub(r"[^a-z\s]", " ", t)
    return [w for w in t.split() if w not in _STOPWORDS and len(w) > 2]


def _conviction_score(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    c = sum(1 for w in tokens if w in _CERTAINTY_WORDS)
    u = sum(1 for w in tokens if w in _UNCERTAINTY_WORDS)
    return (c - u) / len(tokens)


# Data loading

def load_data(data_source: str):
    """Load normalized comments and market metadata for one data source."""
    base_dir     = DATA_ROOTS[data_source]
    comments_csv = os.path.join(base_dir, "comments_clean.csv")
    attrs_clean_csv = os.path.join(base_dir, "attrs_clean.csv")
    markets_csv     = os.path.join(base_dir, "market_attrs.csv")
    prices_dir      = PRICES_DIRS[data_source]

    if not os.path.exists(comments_csv) or os.path.getsize(comments_csv) == 0:
        raise FileNotFoundError(
            f"[load_data] {data_source}: comments file missing or empty: {comments_csv}\n"
            f"  Please copy comments_clean.csv from the data server, or run:\n"
            f"  cd {base_dir} && python clean_comments.py"
        )

    comments = pd.read_csv(
        comments_csv, encoding="latin1" if data_source == "polymarket" else "utf-8-sig",
        engine="python", on_bad_lines="skip"
    )
    # Prefer attrs_clean.csv because it already contains derived topic metadata.
    if os.path.exists(attrs_clean_csv):
        markets = pd.read_csv(attrs_clean_csv)
    else:
        markets = pd.read_csv(markets_csv)

    def _strip_bom(s: str) -> str:
        return s.lstrip('\xef\xbb\xbf').lstrip('\ufeff')
    comments.columns = [_strip_bom(c) for c in comments.columns]
    markets.columns  = [_strip_bom(c) for c in markets.columns]

    if data_source == "polymarket":
        comments, markets = _normalize_polymarket(comments, markets, prices_dir)
    else:
        comments = _normalize_manifold(comments, markets)
        if os.path.isdir(prices_dir):
            comments = _add_prob_at_comment(comments, prices_dir)

    res_map = markets.set_index("market_id")["resolution"].to_dict()
    comments["resolution"] = comments["market_id"].map(res_map)
    return comments, markets


def _normalize_polymarket(comments, markets, prices_dir):
    comments = comments.copy()
    markets  = markets.copy()

    comments["position_side"] = comments["position_side"].str.strip().str.upper()
    markets["resolution"]     = markets["resolution"].str.strip()
    markets = markets[markets["resolution"].isin(["Yes","No"])].copy()
    markets["resolution"] = markets["resolution"].str.upper()

    valid_mids = set(markets["market_id"])
    comments   = comments[comments["market_id"].isin(valid_mids)].reset_index(drop=True)

    if "prob_at_comment" not in comments.columns and os.path.isdir(prices_dir):
        print("  [polymarket] estimating prob_at_comment ...")
        comments = _add_prob_at_comment(comments, prices_dir)
    elif "prob_at_comment" not in comments.columns:
        comments["prob_at_comment"] = float("nan")

    return comments, markets


def _normalize_manifold(comments, markets):
    comments = comments.copy()
    comments["position_side"] = comments["position_side"].fillna("").str.strip().str.upper()
    return comments


def _add_prob_at_comment(comments, prices_dir):
    comments = comments.copy()
    comments["prob_at_comment"] = float("nan")
    comments["_ts"] = (
        pd.to_datetime(comments["created_at"], utc=True, errors="coerce")
        .astype("int64") // 10**9
    )
    for mid in comments["market_id"].unique():
        price_file = os.path.join(prices_dir, f"{mid}.csv")
        if not os.path.exists(price_file):
            continue
        try:
            pr = pd.read_csv(price_file)
        except Exception:
            continue
        yes_pr = pr[pr["outcome"].str.strip().str.lower() == "yes"].sort_values("ts")
        if yes_pr.empty:
            continue
        ts_arr = yes_pr["ts"].to_numpy()
        pr_arr = yes_pr["price"].to_numpy()
        mask   = comments["market_id"] == mid
        idxs   = np.clip(
            np.searchsorted(ts_arr, comments.loc[mask, "_ts"].to_numpy()),
            0, len(ts_arr) - 1
        )
        comments.loc[mask, "prob_at_comment"] = pr_arr[idxs]
    comments.drop(columns=["_ts"], inplace=True)
    return comments


def load_positions(data_source: str) -> pd.DataFrame:
    """Load normalized position snapshots for one data source."""
    path = os.path.join(DATA_ROOTS[data_source], "positions_clean.csv")
    pos  = pd.read_csv(path, engine="python", on_bad_lines="skip")
    pos.columns = [c.strip() for c in pos.columns]
    return pos


def filter_labeled(df: pd.DataFrame) -> pd.DataFrame:
    """Keep comments with explicit YES/NO positions, good text quality, and English text."""
    df = df.copy()
    df = df[df["position_side"].isin(["YES","NO"])]
    df = df[df["text"].apply(is_quality_text)]
    df = df[df["text"].apply(is_english)]
    df = df[df["position_size"].fillna(0) > 0]
    return df.reset_index(drop=True)



def _normalize_mkt_key(s) -> str:
    """Normalize a market key by trimming, lowercasing, and removing an optional 0x prefix."""
    s = str(s).lower().strip()
    if s.startswith("0x"):
        s = s[2:]
    return s


def build_task2_pairs(comments: pd.DataFrame, positions: pd.DataFrame,
                      data_source: str, hold_hours: int = None) -> pd.DataFrame:
    """
    Build Task2 supervision pairs.

    For each labeled comment, look forward within the observation window and infer
    whether the same user later flipped, increased, decreased, or held the position
    in the same market.
    """
    _hold_h = hold_hours if hold_hours is not None else HOLD_HOURS
    HOLD_SECS = _hold_h * 3600

    if data_source == "polymarket":
        c_user, c_mkt = "proxy_wallet", "condition_id"
        p_user, p_mkt = "proxy_wallet", "conditionId"
        pos_ts = positions["timestamp"].astype(float)
    else:
        c_user, c_mkt = "user_address", "market_id"
        p_user, p_mkt = "user_id",      "contract_id"
        pos_ts = (pd.to_datetime(positions["created_at"], utc=True, errors="coerce")
                    .astype("int64") // 10**9)

    # Keep only labeled comments eligible for action supervision.
    cmt = filter_labeled(comments).copy()
    if c_mkt not in cmt.columns:
        print(f"  [!!] {c_mkt} column not found, skipping Task 2")
        return pd.DataFrame(columns=["question","text","position_side","position_size","action"])
    cmt = cmt.dropna(subset=[c_user, c_mkt])
    cmt["_ts"] = (pd.to_datetime(cmt["created_at"], utc=True, errors="coerce")
                    .astype("int64") // 10**9)
    cmt = cmt.dropna(subset=["_ts"])
    cmt["_ts"] = cmt["_ts"].astype(int)

    # Extra normalization and diagnostics for Polymarket joins.
    if data_source == "polymarket":
        print(f"  [diag] positions columns: {list(positions.columns)}")
        # Automatically detect the condition-id column across naming variants.
        pos_mkt_col = None
        for candidate in ["conditionId", "condition_id", "ConditionId", "CONDITION_ID"]:
            if candidate in positions.columns:
                pos_mkt_col = candidate
                break
        if pos_mkt_col is None:
            print(f"  [!!] positions Market key column not found in positions, skipping Task 2")
            return pd.DataFrame(columns=["question","text","position_side","position_size","action"])

        # Normalize hex-like market identifiers before joining.
        cmt["_mkt_key"]   = cmt[c_mkt].apply(_normalize_mkt_key)
        p_mkt_actual      = pos_mkt_col

        # Show a few normalized samples for debugging broken joins.
        pos_sample = positions[pos_mkt_col].dropna().apply(_normalize_mkt_key).unique()[:3].tolist()
        cmt_sample = cmt["_mkt_key"].dropna().unique()[:3].tolist()
        print(f"  [diag] positions['{pos_mkt_col}'] normalized sample: {pos_sample}")
        print(f"  [diag] comments['{c_mkt}'] normalized sample:    {cmt_sample}")

        # Report the overlap size after normalization.
        pos_mkt_set = set(positions[pos_mkt_col].dropna().apply(_normalize_mkt_key))
        cmt_mkt_set = set(cmt["_mkt_key"].dropna())
        overlap_mkt = pos_mkt_set & cmt_mkt_set
        print(f"  [diag] market key overlap: {len(overlap_mkt)} "
              f"(pos={len(pos_mkt_set)}, cmt={len(cmt_mkt_set)})")
        if not overlap_mkt:
            print("  [!!] No market-key overlap after normalization; check condition_id format and source alignment.")
    else:
        cmt["_mkt_key"] = cmt[c_mkt].astype(str)
        p_mkt_actual    = p_mkt

    join_mkt_col = "_mkt_key"

    # Normalize position timestamps and required fields.
    pos = positions.copy()
    pos["_ts"] = pos_ts.values
    pos = pos.dropna(subset=["_ts", p_user, p_mkt_actual, "position_side", "position_size"])
    pos["_ts"]           = pos["_ts"].astype(int)
    pos["position_side"] = pos["position_side"].str.strip().str.upper()
    pos["position_size"] = pos["position_size"].astype(float)

    if data_source == "polymarket":
        pos["_mkt_key"] = pos[p_mkt_actual].apply(_normalize_mkt_key)
    else:
        pos["_mkt_key"] = pos[p_mkt_actual].astype(str)

    # Restrict positions to users and markets that actually appear in comments.
    valid_users   = set(cmt[c_user].astype(str))
    valid_markets = set(cmt[join_mkt_col])
    pos_filtered  = pos[pos[p_user].astype(str).isin(valid_users) &
                        pos["_mkt_key"].isin(valid_markets)]

    if data_source == "polymarket":
        print(f"  [diag] pos rows after user+market filter: {len(pos_filtered)} "
              f"(before: {len(pos)}, valid_markets sample: {list(valid_markets)[:3]})")
        if len(pos_filtered) == 0:
            # Fallback diagnostic: test market-only overlap when user keys do not match.
            pos_mkt_only = pos[pos["_mkt_key"].isin(valid_markets)]
            print(f"  [diag-fallback] pos rows by market-key only: {len(pos_mkt_only)}")
            if len(pos_mkt_only) > 0:
                print("  [!!] Market keys overlap but user keys do not match; check proxy_wallet alignment between comments and positions.")
                # Fallback is for diagnostics only and should not change the GT construction logic.
                pos_filtered = pos_mkt_only
                print("  [INFO] Fallback applied: joining Task2 pairs by market key only.")
            else:
                print("  [!!] No market-key overlap even after fallback; condition_id formats may be inconsistent.")

    pos = pos_filtered

    # Build a sorted lookup table keyed by (user, market).
    pos_dict: dict = defaultdict(list)
    for row in pos[[p_user, "_mkt_key", "_ts", "position_side", "position_size"]].itertuples(index=False):
        pos_dict[(str(row[0]), str(row[1]))].append(
            (int(row[2]), str(row[3]), float(row[4]))
        )
    for key in pos_dict:
        pos_dict[key].sort(key=lambda x: x[0])

    records = []
    for _, row in cmt.iterrows():
        user = str(row[c_user])
        mkt  = str(row[join_mkt_col])   # Use the normalized market key.
        key  = (user, mkt)
        if key not in pos_dict:
            continue

        t0       = int(row["_ts"])
        old_side = str(row["position_side"]).upper().strip()
        old_size = float(row["position_size"])

        window = [(ts, side, size) for ts, side, size in pos_dict[key]
                  if t0 < ts <= t0 + HOLD_SECS]

        if not window:
            action = "hold"
        else:
            _, new_side, new_size = window[-1]
            # Treat a fully exited position as a decrease to zero.
            if new_side == "FLAT":
                action = "decrease"
            elif new_side not in ("YES", "NO"):
                continue
            elif new_side != old_side:
                action = "flip"
            elif new_size > old_size * (1 + ACTION_THR):
                action = "increase"
            elif new_size < old_size * (1 - ACTION_THR):
                action = "decrease"
            else:
                action = "hold"

        # Normalize market_id across numeric and string-style identifiers.
        if "market_id" in row.index and pd.notna(row.get("market_id")):
            try:
                _mid_str = str(int(float(row["market_id"])))
            except (ValueError, TypeError):
                _mid_str = str(row["market_id"])
        else:
            _mid_str = str(row[c_mkt])

        records.append({
            "market_id":     _mid_str,
            "created_at":    row.get("created_at", ""),
            "question":      row["question"],
            "text":          row["text"],
            "position_side": old_side,
            "position_size": old_size,
            "action":        action,
        })

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["market_id","created_at","question","text","position_side","position_size","action"])
    return df



import json as _json

# category_primary to benchmark topic mapping.
# Keep a near 1:1 mapping so topic-level experiments retain fine-grained categories.
_CAT_PRIMARY_TO_TOPIC = {
    "crypto":        "crypto",
    "politics_us":   "politics_us",
    "politics_intl": "politics_intl",
    "sports_nba":    "sports_nba",
    "sports_mlb":    "sports_mlb",
    "sports_other":  "sports_other",
    "tech_ai":       "tech_ai",
    "tech_other":    "tech_other",
    "economics":     "economics",
    "science":       "science",
    "culture":       "culture",
    "weather":       "weather",
    "social":        "social",
    "meta":          "meta",
    "other":         "other",
}

# Keyword fallback table aligned with benchmark topics, used by infer_topic fallback rules.
_TOPIC_KWS = {
    "politics_us":   ["president","election","congress","senate","democrat","republican",
                      "trump","biden","harris","governor","mayor","vote","us election",
                      "white house","supreme court","midterm"],
    "politics_intl": ["geopolit","nato","ukraine","russia","israel","iran","gaza",
                      "middle east","foreign policy","war in","conflict","minister",
                      "parliament","referendum","european","china","taiwan"],
    "crypto":        ["crypto","bitcoin","ethereum","blockchain","defi","nft","token",
                      "btc","eth","solana","coin","web3","binance","altcoin"],
    "sports_nba":    ["nba","basketball","lakers","celtics","warriors","knicks","bulls",
                      "bucks","nets","heat","76ers","nuggets","cavaliers","thunder"],
    "sports_mlb":    ["mlb","baseball","yankees","dodgers","cubs","red sox","world series",
                      "mets","batting","pitcher","home run"],
    "sports_other":  ["nfl","super bowl","soccer","world cup","tennis","golf","boxing",
                      "olympic","formula 1","championship","football","f1","ufc"],
    "tech_ai":       ["openai","gpt","llm","artificial intelligence","machine learning",
                      "ai model","chatgpt","language model","ai safety","agi","gemini",
                      "claude","anthropic","deepmind"],
    "tech_other":    ["elon musk","tesla","spacex","twitter","apple","google","microsoft",
                      "meta ","startup","manifold","software","tech company"],
    "economics":     ["federal reserve","interest rate","inflation","gdp","recession",
                      "fed meeting","fomc","basis point","cpi","treasury","stock market",
                      "yield curve","bank","debt ceiling"],
    "science":       ["science","physics","biology","chemistry","quantum","climate",
                      "superconduct","fusion","research","study","discover","space","nasa"],
    "culture":       ["music","movie","film","grammy","oscar","celebrity","pop culture",
                      "award","song","album","box office","tv show","streaming","netflix"],
    "weather":       ["hurricane","tornado","temperature","snow","rain","weather",
                      "storm","flood","drought","celsius","fahrenheit","typhoon"],
    "social":        ["resolution","personal goal","weight","exercise","habit",
                      "quit","lose","gain","challenge","streak","new year"],
    "meta":          ["nonpredictive","metamarket","metaforecasting","selfresolv",
                      "market of the year","manifold question","daily market"],
}


def infer_topic(row):
    """Infer a benchmark topic from metadata using a staged fallback strategy."""
    q_lower = str(row.get("question", "")).lower()

    # Early blacklist for clearly non-financial markets.
    if any(bl in q_lower for bl in _NONFINANCIAL_BLACKLIST):
        return "__blacklisted__"

    # 1) Prefer the precomputed category_primary field.
    cp = str(row.get("category_primary", "") or "").lower().strip()
    if cp and cp not in ("none", "nan", ""):
        return _CAT_PRIMARY_TO_TOPIC.get(cp, "other")

    # 2) category_validated fallback
    cat = str(row.get("category_validated", "") or "").lower()
    if cat and cat not in ("none", "nan", ""):
        for topic, kws in _TOPIC_KWS.items():
            if any(k in cat for k in kws):
                return topic

    # 3) Manifold group_slugs fallback
    gs = str(row.get("group_slugs", "") or "")
    if gs not in ("nan", "None", ""):
        try:
            slug_text = " ".join(_json.loads(gs)).lower()
            for topic, kws in _TOPIC_KWS.items():
                if any(k in slug_text for k in kws):
                    return topic
        except Exception:
            pass

    # 4) Final fallback: keyword match over the market question.
    for topic, kws in _TOPIC_KWS.items():
        if any(k in q_lower for k in kws):
            return topic

    return "other"


def select_qualified_markets(comments: pd.DataFrame, markets: pd.DataFrame,
                             prices_dir: str = "") -> dict:
    """Select markets with enough valid comments and window structure to run experiments."""
    df = comments.copy()
    df["position_side"] = df["position_side"].fillna("").str.strip().str.upper()
    df = df[df["position_side"].isin(["YES", "NO"])]
    df = df[df["text"].apply(is_quality_text)]
    df = df[df["text"].apply(is_english)]
    df["_ts"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["_ts"])

    lc_map    = _lifecycle_days_from_markets(markets)
    topic_map = {str(r["market_id"]): infer_topic(r) for _, r in markets.iterrows()}

    # [14.7.3] Filter blacklisted markets
    blacklisted_mids = {mid for mid, t in topic_map.items() if t == "__blacklisted__"}
    if blacklisted_mids:
        print(f"  [14.7.3] Filtered {len(blacklisted_mids)} non-financial markets (Wordle/Crossword/Puzzle etc.)")
        df = df[~df["market_id"].astype(str).isin(blacklisted_mids)]

    # Use price CSV as lifecycle fallback
    price_lc: dict = {}
    if prices_dir and os.path.isdir(prices_dir):
        for mid in df["market_id"].unique():
            pf = os.path.join(prices_dir, f"{mid}.csv")
            if not os.path.exists(pf):
                continue
            try:
                pr     = pd.read_csv(pf)
                yes_pr = pr[pr["outcome"].str.strip().str.lower() == "yes"].sort_values("ts")
                if not yes_pr.empty:
                    ts = yes_pr["ts"].to_numpy(float)
                    price_lc[str(mid)] = (ts[-1] - ts[0]) / 86400
            except Exception:
                pass

    result: dict = {}
    for mid, grp in df.groupby("market_id"):
        mid_str = str(mid)
        lifecycle = lc_map.get(mid_str) or price_lc.get(mid_str)
        if not lifecycle or lifecycle <= 0:
            continue

        window_days = max(1, min(30, int(lifecycle // N_WINDOWS_TARGET)))
        window_secs = window_days * 86400
        t_start     = grp["_ts"].min()

        win_idx   = ((grp["_ts"] - t_start).dt.total_seconds() // window_secs).astype(int)
        cnt       = grp.groupby(win_idx).size()
        n_pass    = int((cnt >= MIN_COMMENTS_PER_WINDOW).sum())

        if n_pass >= MIN_WINDOWS_PASS:
            topic = topic_map.get(mid_str, "other")
            if topic == "__blacklisted__":  # extra safety filter
                continue
            result[mid_str] = {
                "topic":          topic,
                "window_days":    window_days,
                "lifecycle_days": lifecycle,
                "n_pass_windows": n_pass,
            }

    return result


def _lifecycle_days_from_markets(markets: pd.DataFrame) -> dict:
    """Extract per-market lifecycle length in days from market metadata."""
    cols = set(markets.columns)
    if "created_time" in cols and "resolution_time" in cols:
        start_col, end_col = "created_time", "resolution_time"
    elif "created_at" in cols and "end_date" in cols:
        start_col, end_col = "created_at", "end_date"
    else:
        return {}

    result = {}
    for _, row in markets.iterrows():
        mid = str(row["market_id"])
        try:
            t_start = pd.to_datetime(row[start_col], utc=True, errors="coerce")
            t_end   = pd.to_datetime(row[end_col],   utc=True, errors="coerce")
            if pd.isna(t_start) or pd.isna(t_end):
                continue
            days = (t_end - t_start).total_seconds() / 86400
            if days > 0:
                result[mid] = days
        except Exception:
            pass
    return result


def build_all_windows(comments: pd.DataFrame, prices_dir: str = "",
                      markets: pd.DataFrame = None) -> list:
    """Build adaptive time windows for Task3 using market lifecycle length."""
    df = comments.copy()
    if "prob_at_comment" not in df.columns:
        df["prob_at_comment"] = float("nan")
    # Note: prob_at_comment NaN rows are skipped per-window below; don't pre-filter here
    # because manifold never populates this field and filtering would empty the DataFrame,
    # causing apply() on an empty object-dtype Series to lose all columns in pandas 1.x.
    df = df[df["resolution"].str.upper().isin(["YES","NO"])]
    df = df[df["position_side"].isin(["YES","NO"])]
    df = df[df["text"].apply(is_quality_text)]
    df = df[df["text"].apply(is_english)]
    df["_ts"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["_ts"])

    meta_lifecycle: dict = _lifecycle_days_from_markets(markets) if markets is not None else {}

    price_cache: dict = {}
    window_days_map: dict = {}
    DEFAULT_WINDOW_DAYS = 7

    if prices_dir and os.path.isdir(prices_dir):
        for mid in df["market_id"].unique():
            pf = os.path.join(prices_dir, f"{str(mid)}.csv")
            if not os.path.exists(pf):
                continue
            try:
                pr     = pd.read_csv(pf)
                yes_pr = pr[pr["outcome"].str.strip().str.lower() == "yes"].sort_values("ts")
                if yes_pr.empty:
                    continue
                ts_arr = yes_pr["ts"].to_numpy(dtype=float)
                pr_arr = yes_pr["price"].to_numpy(dtype=float)
                price_cache[mid] = (ts_arr, pr_arr)
            except Exception:
                pass

    for mid in df["market_id"].unique():
        mid_str = str(mid)
        if mid_str in meta_lifecycle:
            lifecycle_days = meta_lifecycle[mid_str]
        elif mid in price_cache:
            ts_arr = price_cache[mid][0]
            lifecycle_days = (ts_arr[-1] - ts_arr[0]) / 86400
        else:
            lifecycle_days = DEFAULT_WINDOW_DAYS * 8
        window_days_map[mid] = max(1, min(30, int(lifecycle_days // 8)))

    market_windows: dict = defaultdict(list)

    for mid, grp in df.groupby("market_id"):
        grp      = grp.sort_values("_ts").copy()
        question = grp["question"].iloc[0]
        res      = grp["resolution"].iloc[0]

        window_days = window_days_map.get(mid, DEFAULT_WINDOW_DAYS)
        window_secs = window_days * 86400

        if mid in price_cache:
            t_start = pd.Timestamp(price_cache[mid][0][0], unit="s", tz="UTC")
        else:
            t_start = grp["_ts"].min()

        grp["window_idx"] = (
            (grp["_ts"] - t_start).dt.total_seconds() // window_secs
        ).astype(int)

        for widx, dg in grp.groupby("window_idx"):
            if len(dg) < TASK3_MIN_WINDOW_COMMENTS:
                continue
            avg_odds = dg["prob_at_comment"].mean()
            if pd.isna(avg_odds):
                continue

            n_yes = int((dg["position_side"] == "YES").sum())
            n_no  = int((dg["position_side"] == "NO").sum())
            yes_w = dg.loc[dg["position_side"] == "YES", "position_size"].fillna(1).sum()
            no_w  = dg.loc[dg["position_side"] == "NO",  "position_size"].fillna(1).sum()

            total_w = yes_w + no_w
            if yes_w > no_w:
                consensus = "YES"
            elif no_w > yes_w:
                consensus = "NO"
            else:
                consensus = None

            # Consensus strength is measured as directional imbalance.
            strength = abs(yes_w - no_w) / total_w if total_w > 0 else 0.0
            consensus_dir = consensus if strength >= CONSENSUS_STRENGTH_THR else None

            # Share of comments in the window that carry an explicit YES/NO position.
            n_window_total = len(dg)
            pos_comment_frac = (n_yes + n_no) / n_window_total if n_window_total > 0 else 0.0

            win_start_ts = t_start + pd.Timedelta(seconds=int(widx) * window_secs)

            market_windows[mid].append({
                "market_id":            mid,
                "question":             question,
                "resolution":           res,
                "window_idx":           int(widx),
                "window_days":          window_days,
                "win_start_ts":         win_start_ts,
                "avg_odds":             avg_odds,
                "n_yes":                n_yes,
                "n_no":                 n_no,
                "yes_weight":           yes_w,
                "no_weight":            no_w,
                "consensus_strength":   strength,
                "comment_consensus":    consensus,         # Raw binary consensus
                "comment_consensus_dir": consensus_dir,    # Strong-consensus label
                "pos_comment_frac":     pos_comment_frac,  # Share of positioned comments
                "next_window_avg_odds": None,
                "comments": dg[["text","position_side","position_size"]].to_dict("records"),
            })

    all_windows = []
    for mid, wins in market_windows.items():
        wins_sorted = sorted(wins, key=lambda w: w["window_idx"])
        ts_arr, pr_arr = price_cache.get(mid, (None, None))

        for i, w in enumerate(wins_sorted):
            window_secs = w["window_days"] * 86400

            if ts_arr is not None:
                t_next_start = w["win_start_ts"].timestamp() + window_secs
                t_next_end   = t_next_start + window_secs
                mask = (ts_arr >= t_next_start) & (ts_arr < t_next_end)
                if mask.sum() >= 1:
                    w["next_window_avg_odds"] = float(pr_arr[mask].mean())

            if w["next_window_avg_odds"] is None and i + 1 < len(wins_sorted):
                if wins_sorted[i+1]["window_idx"] == w["window_idx"] + 1:
                    w["next_window_avg_odds"] = wins_sorted[i+1]["avg_odds"]

            all_windows.append(w)

    return all_windows


def build_all_windows_comment_count(comments: pd.DataFrame, prices_dir: str = "",
                                    markets: pd.DataFrame = None,
                                    n_groups: int = None) -> list:
    """Build Task3 windows by splitting each market into equal comment-count groups."""
    if n_groups is None:
        n_groups = N_COMMENT_GROUPS

    df = comments.copy()
    if "prob_at_comment" not in df.columns:
        df["prob_at_comment"] = float("nan")
    # Do NOT pre-filter on prob_at_comment here; per-window avg_odds check at line ~1326 handles NaN.
    df = df[df["resolution"].isin(["YES", "NO"])]
    df = df[df["position_side"].isin(["YES", "NO"])]
    df = df[df["text"].apply(is_quality_text)]
    df = df[df["text"].apply(is_english)]
    df["_ts"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["_ts"])

    price_cache: dict = {}
    if prices_dir and os.path.isdir(prices_dir):
        for mid in df["market_id"].unique():
            pf = os.path.join(prices_dir, f"{str(mid)}.csv")
            if not os.path.exists(pf):
                continue
            try:
                pr = pd.read_csv(pf)
                yes_pr = pr[pr["outcome"].str.strip().str.lower() == "yes"].sort_values("ts")
                if yes_pr.empty:
                    continue
                ts_arr = yes_pr["ts"].to_numpy(dtype=float)
                pr_arr = yes_pr["price"].to_numpy(dtype=float)
                price_cache[mid] = (ts_arr, pr_arr)
            except Exception:
                pass

    all_windows = []
    for mid, grp in df.groupby("market_id"):
        grp = grp.sort_values("_ts").reset_index(drop=True).copy()
        question = grp["question"].iloc[0]
        res = grp["resolution"].iloc[0]
        total_n = len(grp)

        # Require at least three comments per group.
        if total_n < n_groups * 3:
            continue

        group_size = total_n // n_groups
        grp["_group_idx"] = grp.index // group_size
        grp.loc[grp["_group_idx"] >= n_groups, "_group_idx"] = n_groups - 1

        group_data = []
        for gidx, dg in grp.groupby("_group_idx"):
            if len(dg) < 3:
                continue
            avg_odds = dg["prob_at_comment"].mean()
            if pd.isna(avg_odds):
                continue

            n_yes = int((dg["position_side"] == "YES").sum())
            n_no = int((dg["position_side"] == "NO").sum())
            yes_w = dg.loc[dg["position_side"] == "YES", "position_size"].fillna(1).sum()
            no_w = dg.loc[dg["position_side"] == "NO", "position_size"].fillna(1).sum()
            total_w = yes_w + no_w

            if yes_w > no_w:
                consensus = "YES"
            elif no_w > yes_w:
                consensus = "NO"
            else:
                consensus = None

            strength = abs(yes_w - no_w) / total_w if total_w > 0 else 0.0
            consensus_dir = consensus if strength >= CONSENSUS_STRENGTH_THR else None
            pos_comment_frac = (n_yes + n_no) / len(dg) if len(dg) > 0 else 0.0

            group_data.append({
                "market_id":             mid,
                "question":              question,
                "resolution":            res,
                "window_idx":            int(gidx),
                "window_type":           "comment_count",
                "group_size_target":     group_size,
                "n_comments":            len(dg),
                "avg_odds":              avg_odds,
                "n_yes":                 n_yes,
                "n_no":                  n_no,
                "yes_weight":            yes_w,
                "no_weight":             no_w,
                "consensus_strength":    strength,
                "comment_consensus":     consensus,
                "comment_consensus_dir": consensus_dir,
                "pos_comment_frac":      pos_comment_frac,
                "next_window_avg_odds":  None,
                "comments":              dg[["text", "position_side", "position_size"]].to_dict("records"),
            })

        for i in range(len(group_data) - 1):
            group_data[i]["next_window_avg_odds"] = group_data[i + 1]["avg_odds"]

        if group_data and mid in price_cache:
            last_grp_rows = grp[grp["_group_idx"] == grp["_group_idx"].max()]
            if len(last_grp_rows) > 0:
                last_ts = last_grp_rows["_ts"].max().timestamp()
                ts_arr, pr_arr = price_cache[mid]
                after_mask = ts_arr > last_ts
                if after_mask.sum() >= 1:
                    group_data[-1]["next_window_avg_odds"] = float(pr_arr[after_mask][:10].mean())

        all_windows.extend(group_data)

    return all_windows



def run_phase0(df_labeled: pd.DataFrame, task2_pairs: pd.DataFrame,
               all_windows: list, data_source: str) -> dict:
    print(f"\n{'=' * 62}")
    print(f"[{data_source.upper()}] Phase 0: descriptive validation without LLM")
    print(f"{'=' * 62}")
    r12 = _phase0_p1_p2(df_labeled, data_source)
    r3  = _phase0_p3_bpla(all_windows, data_source)
    r4  = _phase0_p4_task2(task2_pairs, data_source)
    return {**r12, **r3, **r4}


def _phase0_p1_p2(df_labeled: pd.DataFrame, data_source: str) -> dict:
    df = df_labeled.copy()
    n_yes = int((df.position_side == "YES").sum())
    n_no  = int((df.position_side == "NO").sum())
    print(f"\n[P1] Naive Bayes 5-fold CV  (YES={n_yes}, NO={n_no}, total={len(df)})")

    tokens_list = [_tokenize(str(t)) for t in df["text"]]
    indices = list(range(len(df)))
    random.shuffle(indices)
    k, fold_sz = 5, len(indices) // 5
    folds = [indices[i*fold_sz:(i+1)*fold_sz] for i in range(k)]

    total_correct = total_n = 0
    for fi in range(k):
        test_set  = set(folds[fi])
        train_idx = [i for i in indices if i not in test_set]
        yes_wf: Counter = Counter()
        no_wf:  Counter = Counter()
        n_y_tr = n_n_tr = 0
        for i in train_idx:
            side = df.iloc[i]["position_side"]
            if side == "YES":
                yes_wf.update(tokens_list[i]); n_y_tr += 1
            else:
                no_wf.update(tokens_list[i]);  n_n_tr += 1
        V     = len(set(yes_wf) | set(no_wf))
        tot_y = sum(yes_wf.values()) + V
        tot_n = sum(no_wf.values())  + V
        lp_y  = math.log((n_y_tr + 1) / (n_y_tr + n_n_tr + 2))
        lp_n  = math.log((n_n_tr + 1) / (n_y_tr + n_n_tr + 2))
        for i in folds[fi]:
            toks = tokens_list[i]
            if not toks:
                continue
            sy = lp_y + sum(math.log((yes_wf.get(w,0)+1)/tot_y) for w in toks)
            sn = lp_n + sum(math.log((no_wf.get(w, 0)+1)/tot_n) for w in toks)
            pred = "YES" if sy > sn else "NO"
            total_correct += (pred == df.iloc[i]["position_side"])
            total_n += 1

    acc   = total_correct / max(total_n, 1)
    p_bin = binom_pvalue_greater(total_correct, total_n, 0.5)
    print(f"  NB 5-fold CV: acc={acc:.4f}  ({total_correct}/{total_n})")
    print(f"  Binomial test p: {p_bin:.2e}  "
          f"{'[OK] Text signal significant' if p_bin < 0.05 else '[!!] Not significant'}")

    print(f"\n[P2] Costly Signal — Spearman(log_size, |conviction|)")
    df2 = df.copy()
    df2["conv_abs"] = [abs(_conviction_score(str(t))) for t in df2["text"]]
    df2 = df2[df2["position_size"].notna() & (df2["position_size"] > 0)].copy()
    df2["log_size"] = np.log1p(df2["position_size"].astype(float))
    rho, p_rho = spearman_rho(df2["log_size"].tolist(), df2["conv_abs"].tolist())
    print(f"  Spearman ρ={rho:.4f}  p={p_rho:.4f}  n={len(df2)}")
    print(f"  {'[OK] Larger positions are associated with stronger conviction text' if rho > 0 and p_rho < 0.05 else '[!!] No significant monotonic relation found'}")

    q25 = float(df2["position_size"].quantile(0.25))
    q75 = float(df2["position_size"].quantile(0.75))
    small = df2[df2["position_size"] <= q25]
    large = df2[df2["position_size"] >= q75]
    print(f"  Small positions (<=p25, med=${small['position_size'].median():,.0f}): "
          f"mean|conv|={small['conv_abs'].mean():.5f}  n={len(small)}")
    print(f"  Large positions (>=p75, med=${large['position_size'].median():,.0f}): "
          f"mean|conv|={large['conv_abs'].mean():.5f}  n={len(large)}")
    mwu_p = mannwhitney_pvalue_greater(large["conv_abs"].tolist(), small["conv_abs"].tolist())
    print(f"  MWU (large>small): p={mwu_p:.4f}  {'[OK]' if mwu_p<0.05 else '[!!]'}")

    return {"p1_acc": acc, "p1_p": p_bin, "p1_n": total_n,
            "p2_rho": rho, "p2_p": p_rho, "p2_mwu": mwu_p}


def _phase0_p3_bpla(all_windows: list, data_source: str) -> dict:
    print(f"\n[P3] Task 3' BPLA -- weighted consensus vs. next-window price direction")

    # Windows with a consensus label and a valid next-window price.
    valid = [w for w in all_windows
             if w["comment_consensus"] is not None
             and w["next_window_avg_odds"] is not None]
    print(f"  Total windows: {len(all_windows)}  Valid (binary consensus, next price): {len(valid)}")

    if len(valid) < 10:
        print("  [!!] Not enough valid windows for BPLA; skipping.")
        return {"p3_total": 0, "p3_bpla_3c": float("nan"), "p3_bpla_3c_p": 1.0,
                "p3_bpla_2c": float("nan"), "p3_bpla_2c_p": 1.0, "p3_bpla_2c_n": 0}

    records_3c = []  # All windows; NEUTRAL counts as an error for 3-class stats.
    records_2c = []  # Strong-consensus windows only.
    for w in valid:
        delta = w["next_window_avg_odds"] - w["avg_odds"]
        if abs(delta) <= PRICE_EPS:
            continue
        gt_dir   = "UP" if delta > 0 else "DOWN"
        strength = w.get("consensus_strength", 1.0)  # Backward-compatible default

        consensus     = w["comment_consensus"]
        consensus_dir = w.get("comment_consensus_dir")
        stat_bin = "UP" if consensus == "YES" else "DOWN"
        stat_dir = "UP" if consensus_dir == "YES" else ("DOWN" if consensus_dir == "NO" else "NEUTRAL")
        hit_3c = (stat_bin == gt_dir)                   # 3c statistic with no NEUTRAL label
        hit_2c = (stat_dir != "NEUTRAL") and (stat_dir == gt_dir)

        records_3c.append({"stat": stat_bin, "gt": gt_dir, "correct": hit_3c,
                            "strength": strength, "avg_odds": w["avg_odds"]})
        if stat_dir != "NEUTRAL":
            records_2c.append({"stat": stat_dir, "gt": gt_dir, "correct": hit_2c,
                                "strength": strength})

    n_3c = len(records_3c)
    n_2c = len(records_2c)
    bpla_3c = sum(r["correct"] for r in records_3c) / n_3c if n_3c > 0 else float("nan")
    bpla_2c = sum(r["correct"] for r in records_2c) / n_2c if n_2c > 0 else float("nan")
    p_3c = binom_pvalue_greater(int(sum(r["correct"] for r in records_3c)), n_3c) if n_3c > 0 else 1.0
    p_2c = binom_pvalue_greater(int(sum(r["correct"] for r in records_2c)), n_2c) if n_2c > 0 else 1.0
    cov  = n_2c / n_3c if n_3c > 0 else 0.0

    n_down  = sum(1 for r in records_3c if r["gt"] == "DOWN")
    n_up    = sum(1 for r in records_3c if r["gt"] == "UP")
    bpla_naive = n_down / n_3c if n_3c > 0 else float("nan")  # Naive baseline: always predict DOWN
    bpla_above_naive = bpla_3c - bpla_naive if n_3c > 0 else float("nan")

    sub_up   = [r for r in records_3c if r["gt"] == "UP"]
    sub_down = [r for r in records_3c if r["gt"] == "DOWN"]
    bpla_up   = sum(r["correct"] for r in sub_up)   / len(sub_up)   if sub_up   else float("nan")
    bpla_down = sum(r["correct"] for r in sub_down) / len(sub_down) if sub_down else float("nan")

    print(f"  Significant price-change windows (|delta|>{PRICE_EPS}): {n_3c}  (UP={n_up}, DOWN={n_down})")
    print(f"  [3c] BPLA_stat(all)      = {bpla_3c:.4f}  p={p_3c:.4f}  "
          f"{'[OK]' if p_3c < 0.10 else '[!!]'}")
    print(f"  [2c] BPLA_stat(strong-consensus) = {bpla_2c:.4f}  p={p_2c:.4f}  "
          f"coverage={cov:.1%}({n_2c}/{n_3c})  {'[OK]' if p_2c < 0.10 else '[!!]'}")
    print(f"  (strong-consensus threshold: strength >= {CONSENSUS_STRENGTH_THR})")
    print(f"  [v7-B] naive_DOWNbaseline    = {bpla_naive:.4f}  above_naive = {bpla_above_naive:+.4f}  "
          f"{'[above-naive-baseline]' if bpla_above_naive > 0 else '[!!below-naive]'}")
    print(f"  [v7-B] stratified: UP_acc={bpla_up:.3f}(n={n_up})  DOWN_acc={bpla_down:.3f}(n={n_down})")

    return {"p3_total": n_3c,
            "p3_bpla_3c": bpla_3c, "p3_bpla_3c_p": p_3c,
            "p3_bpla_2c": bpla_2c, "p3_bpla_2c_p": p_2c, "p3_bpla_2c_n": n_2c,
            "p3_bpla_naive": bpla_naive, "p3_bpla_above_naive": bpla_above_naive,
            "p3_bpla_up": bpla_up, "p3_bpla_down": bpla_down}


def _phase0_p4_task2(task2_pairs: pd.DataFrame, data_source: str) -> dict:
    print(f"\n[P4] Task 2 action distribution statistics")
    if task2_pairs.empty:
        print("  [!!] No valid Task2 pairs; skipping.")
        return {"p4_total": 0}

    total = len(task2_pairs)
    dist  = task2_pairs["action"].value_counts()
    print(f"  Valid (comment, action) pairs: {total}")
    for act in ["flip","increase","decrease","hold"]:
        n = dist.get(act, 0)
        print(f"    {act:<10}: {n:5d}  ({n/total:.1%})")

    print(f"  Avg position size by action:")
    for act in ["flip","increase","decrease","hold"]:
        sub = task2_pairs[task2_pairs["action"] == act]
        if len(sub):
            print(f"    {act:<10}: ${sub['position_size'].median():>10,.0f} (median)  n={len(sub)}")

    return {"p4_total": total,
            "p4_flip":     dist.get("flip",0),
            "p4_increase": dist.get("increase",0),
            "p4_decrease": dist.get("decrease",0),
            "p4_hold":     dist.get("hold",0)}



def parse_yes_no(raw: str) -> str:
    up = raw.strip().upper()
    for tok in ["YES","NO","NEUTRAL"]:
        if up.startswith(tok) or f" {tok}" in up or f"\n{tok}" in up:
            return tok
    if "YES" in up: return "YES"
    if "NO"  in up: return "NO"
    return "NEUTRAL"


def parse_up_down(raw: str) -> str:
    up = raw.strip().upper()
    for tok in ["UP","DOWN","NEUTRAL"]:
        if up.startswith(tok) or f" {tok}" in up or f"\n{tok}" in up:
            return tok
    if "UP"   in up: return "UP"
    if "DOWN" in up: return "DOWN"
    return "NEUTRAL"


def parse_action(raw: str) -> str:
    up = raw.strip().upper()
    for act in ["FLIP","INCREASE","DECREASE","HOLD"]:
        if up.startswith(act) or f" {act}" in up or f"\n{act}" in up:
            return act.lower()
    for act in ["FLIP","INCREASE","DECREASE","HOLD"]:
        if act in up:
            return act.lower()
    return "UNKNOWN"



def run_task0(comments: pd.DataFrame, data_source: str, out_subdir: str, n_each: int = 100):
    """Run Task0 in one-shot and zero-shot mode and return the explicit-position pool."""
    _log(f"\n{'='*62}")
    _log(f"[{data_source.upper()}] Phase1 Task0 — Position Detection (1-shot & 0-shot)")
    _log(f"{'='*62}")

    df = comments[comments["text"].apply(is_quality_text) &
                  comments["text"].apply(is_english)].copy()

    # Keep only positioned comments with explicit position language for the narrow T0 setting.
    all_pos  = df[df["position_side"].isin(["YES","NO"]) & (df["position_size"].fillna(0) > 0)]
    pos_pool_raw = all_pos[all_pos["text"].apply(has_explicit_pos_signal)].copy()

    # Sports topics can override the explicit-signal threshold.
    _topic_key = data_source.split("/")[-1] if "/" in data_source else ""
    _thr = 0 if _topic_key in SPORTS_TOPICS_ZERO_THR else EXPLICIT_QUALITY_THR

    pos_pool_raw["_explicit_score"] = pos_pool_raw["text"].apply(_score_explicit_signal)
    score_dist = pos_pool_raw["_explicit_score"].value_counts().sort_index().to_dict()
    _log(f"  T0 explicit signal quality distribution(0-4): {score_dist}  threshold>={_thr}"
         f"{'(sports topic: threshold forced to 0, no filtering)' if _topic_key in SPORTS_TOPICS_ZERO_THR else ''}")
    pos_pool = pos_pool_raw[pos_pool_raw["_explicit_score"] >= _thr].copy()
    pos_pool = pos_pool.drop(columns=["_explicit_score"], errors="ignore")

    neg_pool = df[~df["position_side"].isin(["YES","NO"])]

    explicit_rate = len(pos_pool) / max(len(all_pos), 1)
    _log(
        f"  pos_pool explicit coverage (quality >= EXPLICIT_QUALITY_THR): "
        f"{len(pos_pool)}/{len(all_pos)} ({explicit_rate:.1%})  neg_pool={len(neg_pool)}"
    )
    _log(f"  [funnel] T0 explicit_pos_pool -> {len(pos_pool)} rows passed to Task1 HP")

    available_each = min(len(pos_pool), len(neg_pool))
    if available_each < MIN_EXPLICIT_TASK0:
        _log(f"  [!!] Sample insufficient (with_pos+explicit={len(pos_pool)}, neg_pool={len(neg_pool)}, threshold={MIN_EXPLICIT_TASK0})", "warning")
        return None

    effective_n_each = available_each if RUN_FULL else min(n_each, available_each)
    pos_s = _sample_df(pos_pool, None if RUN_FULL else effective_n_each); pos_s["has_position"] = True
    neg_s = _sample_df(neg_pool, None if RUN_FULL else effective_n_each); neg_s["has_position"] = False
    sample = pd.concat([pos_s, neg_s]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    if RUN_FULL:
        _log(f"Full run: using all eligible Task0 rows with_position={len(pos_s)}, without_position={len(neg_s)}, total={len(sample)}")
    else:
        _log(f"Sampling: with_position={effective_n_each}, without_position={effective_n_each}, total={len(sample)} (shared by 1-shot and 0-shot)")

    df_1shot = _run_task0_loop(sample, data_source, out_subdir, "1shot", prompt_task0)
    df_0shot = _run_task0_loop(sample, data_source, out_subdir, "0shot", prompt_task0_zeroshot)

    _report_task0_compare(df_1shot, df_0shot, data_source)
    return {"pos_pool": pos_pool}


def _run_task0_loop(sample: pd.DataFrame, data_source: str, out_subdir: str,
                    variant: str, prompt_fn) -> pd.DataFrame:
    """Internal Task0 loop with checkpoint/resume support."""
    csv_path  = os.path.join(out_subdir, f"qwen3_{data_source.replace('/', '_')}_task0_{variant}.csv")
    state_key = f"{MODEL}/{data_source}/task0_{variant}"
    expected  = len(sample)

    existing = _csv_load_partial(csv_path)
    if len(existing) > 0 and "market_id" in existing.columns and "created_at" in existing.columns:
        done_keys = set(zip(existing["market_id"].astype(str), existing["created_at"].astype(str)))
    else:
        done_keys = set()
    results  = existing.to_dict("records") if len(existing) > 0 else []

    if _is_complete(csv_path, expected) or _state_is_done(state_key):
        _log(f"  [RESUME] Task0-{variant} {data_source} already complete ({len(existing)} rows), skipping LLM calls")
        _report_task0(existing, data_source, variant)
        return existing

    if done_keys:
        _log(f"  [RESUME] Task0-{variant} {data_source} partial progress ({len(done_keys)}/{expected}), continuing")

    _log(f"\n[Task0-{variant}] start inference ({len(done_keys)}/{expected} already done)")

    def _t0_worker(row, _variant=variant):
        pred_raw = query_llm(prompt_fn(row["question"], row["text"]))
        up = pred_raw.strip().upper()
        pred = True  if (up.startswith("YES") or " YES" in up) else \
               False if (up.startswith("NO")  or " NO"  in up) else None
        gt      = row["has_position"]
        correct = (pred == gt) if pred is not None else False
        print(f"  [{_variant}] GT={'POS' if gt else 'NEG'} "
              f"PRED={'POS' if pred else 'NEG' if pred is not None else 'N/A'} "
              f"{'[Y]' if correct else '[N]'}  {str(row['text'])[:48]!r}")
        return {"market_id": row.get("market_id", ""), "created_at": row.get("created_at", ""),
                "question": str(row.get("question", ""))[:200],
                "text": str(row["text"])[:120],
                "gt": gt, "pred": pred, "correct": correct,
                "raw_response": str(pred_raw)[:300],
                "model": MODEL}

    sample_rows = [((str(r.get("market_id", "")), str(r.get("created_at", ""))), r)
                   for _, r in sample.iterrows()]
    results = _run_infer_loop(sample_rows, done_keys, _t0_worker, results, csv_path, expected)

    df_res = pd.DataFrame(results)
    df_res.to_csv(csv_path, index=False)
    _state_mark_done(state_key)
    _report_task0(df_res, data_source, variant)
    return df_res


def _report_task0(df_res, data_source: str, variant: str = ""):
    if isinstance(df_res, list):
        df_res = pd.DataFrame(df_res)
    if len(df_res) == 0:
        return
    df_res = _fix_bool_cols(df_res)
    total   = len(df_res)
    correct = int(df_res["correct"].sum())
    acc     = correct / total
    p_bin   = binom_pvalue_greater(correct, total)
    pos_sub = df_res[df_res["gt"] == True]
    neg_sub = df_res[df_res["gt"] == False]
    recall  = pos_sub["correct"].mean() if len(pos_sub) else float("nan")
    spec    = neg_sub["correct"].mean() if len(neg_sub) else float("nan")
    pred_yes  = df_res[df_res["pred"] == True]
    precision = pred_yes["gt"].mean() if len(pred_yes) else float("nan")
    pred_dist = {"YES": int((df_res["pred"] == True).sum()),
                 "NO":  int((df_res["pred"] == False).sum()),
                 "N/A": int(df_res["pred"].isna().sum())}
    tag = f"[{variant}] " if variant else ""
    _log(f"\n------ [{data_source.upper()}] Task0 {tag}------")
    _log(f"  Total samples:{total}  acc={acc:.3f}  p={p_bin:.4f}  "
         f"{'[OK]' if p_bin<0.05 else '[!!]'}")
    _log(f"  Recall(HP->YES)={recall:.3f}  Specificity(NP->NO)={spec:.3f}  Precision={precision:.3f}")
    _log(f"  Prediction distribution: {pred_dist}")
    logger.info(f"RESULT task0-{variant} {data_source}: acc={acc:.3f} recall={recall:.3f} precision={precision:.3f}")


def _report_task0_compare(df_1shot, df_0shot, data_source: str):
    """Compare Task0 metrics between the one-shot and zero-shot settings."""
    if df_1shot is None or df_0shot is None:
        return
    if isinstance(df_1shot, list): df_1shot = pd.DataFrame(df_1shot)
    if isinstance(df_0shot, list): df_0shot = pd.DataFrame(df_0shot)
    if len(df_1shot) == 0 or len(df_0shot) == 0:
        return

    def _metrics(df):
        acc  = df["correct"].mean()
        rec  = df[df["gt"]==True]["correct"].mean() if (df["gt"]==True).any() else float("nan")
        spec = df[df["gt"]==False]["correct"].mean() if (df["gt"]==False).any() else float("nan")
        pyes = df[df["pred"]==True]
        prec = pyes["gt"].mean() if len(pyes) > 0 else float("nan")
        return acc, rec, spec, prec

    a1, r1, s1, p1 = _metrics(df_1shot)
    a0, r0, s0, p0 = _metrics(df_0shot)

    _log(f"\n------ [{data_source.upper()}] Task0 1-shot vs 0-shot comparison ------")
    _log(f"  {'':15s}  {'1-shot':>8}  {'0-shot':>8}  {'Δ(1-0)':>8}")
    _log(f"  {'acc':15s}  {a1:8.3f}  {a0:8.3f}  {a1-a0:+8.3f}")
    _log(f"  {'recall':15s}  {r1:8.3f}  {r0:8.3f}  {r1-r0:+8.3f}")
    _log(f"  {'specificity':15s}  {s1:8.3f}  {s0:8.3f}  {s1-s0:+8.3f}")
    _log(f"  {'precision':15s}  {p1:8.3f}  {p0:8.3f}  {p1-p0:+8.3f}")


def run_task1(df_labeled: pd.DataFrame, data_source: str, out_subdir: str,
              n_each: int = 200, comments_nopos: pd.DataFrame = None,
              explicit_pos_pool: pd.DataFrame = None):
    """Run Task1 on both positioned and non-positioned comment groups."""
    _log(f"\n{'='*62}")
    _log(f"[{data_source.upper()}] Phase1 Task1 — Belief Inference (has_pos / no_pos split)")
    _log(f"{'='*62}")
    # V14: extract topic slug for topic-aware prompt injection (crypto sarcasm note).
    _topic_key = data_source.split("/")[-1] if "/" in data_source else ""

    csv_path  = os.path.join(out_subdir, f"qwen3_{data_source.replace('/', '_')}_task1.csv")
    state_key = f"{MODEL}/{data_source}/task1"

    _use_explicit = False
    if explicit_pos_pool is not None and len(explicit_pos_pool) >= 2 * MIN_EXPLICIT_TASK0:
        ep_yes = explicit_pos_pool[explicit_pos_pool["position_side"] == "YES"]
        ep_no  = explicit_pos_pool[explicit_pos_pool["position_side"] == "NO"]
        if len(ep_yes) >= MIN_EXPLICIT_TASK0 and len(ep_no) >= MIN_EXPLICIT_TASK0:
            yes_pool = ep_yes.copy()
            no_pool  = ep_no.copy()
            _use_explicit = True
    if not _use_explicit:
        yes_pool = df_labeled[df_labeled["position_side"] == "YES"].copy()
        no_pool  = df_labeled[df_labeled["position_side"] == "NO"].copy()
    hp_source = "T0 explicit_pos_pool" if _use_explicit else "full df_labeled fallback"
    _log(f"  [funnel] Task1 HP source: {hp_source}  YES={len(yes_pool)}  NO={len(no_pool)}")
    yes_n = len(yes_pool) if RUN_FULL else min(n_each, len(yes_pool))
    no_n  = len(no_pool) if RUN_FULL else min(n_each, len(no_pool))
    yes_s = _sample_df(yes_pool, None if RUN_FULL else yes_n)
    no_s  = _sample_df(no_pool, None if RUN_FULL else no_n)
    haspos = pd.concat([yes_s, no_s])
    haspos["_gt"]          = haspos["position_side"]
    haspos["has_position"] = True

    nopos_parts = []
    if comments_nopos is not None and len(comments_nopos) > 0:
        nopos_pool = comments_nopos[
            ~comments_nopos["position_side"].isin(["YES","NO"]) &
            comments_nopos["text"].apply(is_quality_text) &
            comments_nopos["text"].apply(is_english) &
            comments_nopos["resolution"].isin(["YES","NO"])
        ].copy()
        nopos_n = len(nopos_pool) if RUN_FULL else min(n_each, len(nopos_pool))
        if nopos_n > 0:
            nopos_s = _sample_df(nopos_pool, None if RUN_FULL else nopos_n)
            nopos_s["_gt"]          = nopos_s["resolution"]
            nopos_s["has_position"] = False
            if "position_size" not in nopos_s.columns:
                nopos_s["position_size"] = 0.0
            nopos_parts.append(nopos_s)
            _log(f"  no-position group: n={nopos_n} (GT=resolution, pool={len(nopos_pool)})")
        else:
            _log(f"  no-position group: empty pool, skipping (comments_nopos={len(comments_nopos)})")
    else:
        _log("  No-position group: comments_nopos not provided, skipping")

    sample_parts = [haspos] + nopos_parts
    sample   = pd.concat(sample_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    expected = len(sample)
    _log(f"  With-pos: {len(haspos)}  No-pos: {sum(len(p) for p in nopos_parts)}  Total={expected}")

    existing = _csv_load_partial(csv_path)
    if len(existing) > 0 and "market_id" in existing.columns and "created_at" in existing.columns:
        done_keys = set(zip(existing["market_id"].astype(str), existing["created_at"].astype(str)))
    else:
        done_keys = set()
    results  = existing.to_dict("records") if len(existing) > 0 else []

    if _is_complete(csv_path, expected) or _state_is_done(state_key):
        _log(f"  [RESUME] Task1 {data_source} already complete ({len(existing)} rows), skipping LLM calls")
        logger.info(f"SKIP task1 {data_source} — {len(existing)} rows exist")
        _report_task1(existing, data_source)
        _report_task1_commitment_validity(existing, data_source, out_subdir)
        return existing

    if done_keys:
        _log(f"  [RESUME] Task1 {data_source} partial progress ({len(done_keys)}/{expected}), continuing")
        logger.info(f"RESUME task1 {data_source}: {len(done_keys)} rows done")

    if RUN_FULL:
        _log(f"Full run: {expected} Task1 rows ({len(done_keys)} already done)")
    else:
        _log(f"Sampling: {expected} rows ({len(done_keys)} already done)")

    def _t1_worker(row, _topic=_topic_key):
        _raw_t1 = query_llm(prompt_task1(row["question"], row["text"], topic=_topic))
        pred    = parse_yes_no(_raw_t1)
        gt      = row["_gt"]
        correct = (pred == gt)
        hp_tag  = "HP" if row["has_position"] else "NP"
        print(f"  [{hp_tag}] GT={gt:<3} PRED={pred:<7} {'[Y]' if correct else '[N]'}  "
              f"${row.get('position_size', 0):,.0f}  {str(row['text'])[:44]!r}")
        return {"market_id": row.get("market_id", ""), "created_at": row.get("created_at", ""),
                "user_address": str(row.get("user_address", "") or ""),
                "question": str(row.get("question", ""))[:200],
                "text": str(row["text"])[:120],
                "gt": gt, "pred": pred, "correct": correct,
                "has_position": bool(row["has_position"]),
                "position_size": float(row.get("position_size", 0) or 0),
                "resolution": str(row.get("resolution", "") or "").strip().upper(),
                "raw_response": str(_raw_t1)[:300],
                "model": MODEL}

    sample_rows = [((str(r.get("market_id", "")), str(r.get("created_at", ""))), r)
                   for _, r in sample.iterrows()]
    results = _run_infer_loop(sample_rows, done_keys, _t1_worker, results, csv_path, expected)

    df_res = pd.DataFrame(results)
    df_res.to_csv(csv_path, index=False)
    _state_mark_done(state_key)
    _report_task1(df_res, data_source)
    _report_task1_commitment_validity(df_res, data_source, out_subdir)
    return df_res


def _report_task1(df, data_source: str):
    """Report Task1 metrics in both 3-class and directed 2-class form."""
    if isinstance(df, list):
        df = pd.DataFrame(df)
    if len(df) == 0:
        return
    df = _fix_bool_cols(df)

    total   = len(df)
    correct = int(df["correct"].sum())

    acc_3c       = correct / total
    p_3c         = binom_pvalue_greater(correct, total)
    neutral_rate = (df["pred"] == "NEUTRAL").mean()

    df_dir     = df[df["pred"].isin(["YES","NO"])]
    n_directed = len(df_dir)
    da_correct = int(df_dir["correct"].sum())
    acc_2c     = df_dir["correct"].mean() if n_directed > 0 else float("nan")
    cov        = n_directed / total
    p_2c       = binom_pvalue_greater(da_correct, n_directed) if n_directed > 1 else 1.0

    print(f"\n------ [{data_source.upper()}] Task1 ------")
    print(f"  Total samples:{total}  Prediction distribution: {dict(Counter(df['pred']))}")
    print(f"  NEUTRAL rate: {neutral_rate:.1%}  (NEUTRAL always wrong; reflects LLM avoidance)")
    print(f"  [3c] acc={acc_3c:.3f}  p={p_3c:.4f}  "
          f"{'[OK]' if p_3c<0.05 else '[!!]'}  (strict 3-class, NEUTRAL counted as wrong)")
    print(f"  [2c] DA ={acc_2c:.3f}  p={p_2c:.4f}  coverage={cov:.1%}  n={n_directed}  "
          f"{'[OK]' if p_2c<0.05 else '[!!]'}  (excludes NEUTRAL; directed prediction accuracy)")

    if "has_position" in df.columns:
        print(f"  |-------- Position group comparison (verify position effect on Task1)|----------------------------------------------")
        for grp_label, grp_mask in [("HP (positioned)", df["has_position"]==True),
                                     ("NP (no position)", df["has_position"]==False)]:
            grp = df[grp_mask]
            if len(grp) == 0:
                continue
            grp_dir = grp[grp["pred"].isin(["YES","NO"])]
            g_acc_3c = grp["correct"].mean()
            g_acc_2c = grp_dir["correct"].mean() if len(grp_dir) > 0 else float("nan")
            g_neutral = (grp["pred"] == "NEUTRAL").mean()
            g_cov    = len(grp_dir) / len(grp)
            g_p_2c   = binom_pvalue_greater(int(grp_dir["correct"].sum()), len(grp_dir)) \
                       if len(grp_dir) > 1 else 1.0
            print(f"  └ [{grp_label}] n={len(grp):4d}  "
                  f"acc_3c={g_acc_3c:.3f}  acc_2c={g_acc_2c:.3f}  "
                  f"p_2c={g_p_2c:.4f}  NEUTRAL={g_neutral:.1%}  cov={g_cov:.1%}")
        print(f"  |--------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    df_haspos     = df[df["has_position"] == True] if "has_position" in df.columns else df
    df_haspos_dir = df_haspos[df_haspos["pred"].isin(["YES", "NO"])]
    if len(df_haspos_dir) >= 30:
        q33 = df_haspos_dir["position_size"].quantile(0.33)
        q67 = df_haspos_dir["position_size"].quantile(0.67)
        small  = df_haspos_dir[df_haspos_dir["position_size"] <= q33]
        medium = df_haspos_dir[(df_haspos_dir["position_size"] > q33) & (df_haspos_dir["position_size"] <= q67)]
        large  = df_haspos_dir[df_haspos_dir["position_size"] > q67]
        for lbl, grp in [("small(bot33%)",small),("mid(mid33%)",medium),("large(top33%)",large)]:
            if len(grp):
                print(f"  {lbl}: acc={grp['correct'].mean():.3f}  "
                      f"n={len(grp)}  med=${grp['position_size'].median():,.0f}")
        if len(large) >= 20 and len(small) >= 20:
            u_p = mannwhitney_pvalue_greater(
                large["correct"].astype(int).tolist(),
                small["correct"].astype(int).tolist())
            print(f"  CSD(large-small)={large['correct'].mean()-small['correct'].mean():.3f}  "
                  f"MWU p={u_p:.4f}  {'[OK]' if u_p<0.05 else '[!!]'}")


def _report_task1_commitment_validity(
    df: pd.DataFrame,
    data_source: str,
    out_subdir: str,
) -> dict:
    """Commitment validity experiments on Task1 output.

    Experiment 1 — Wrong-Bettor Test
        Among HP comments where position_side ≠ resolution (i.e. the commenter
        was financially wrong about the market), does the LLM still predict the
        commenter's position_side more often than the market resolution?
        If yes → the LLM reads commitment-shaped language, NOT just correct-prediction
        language. This rules out the "smart-money" alternative explanation.

    Experiment 2 — Position-Size Dose-Response
        Divide HP directional predictions (non-NEUTRAL) into four position-size
        quartiles. If commitment drives language, T1 DA should increase monotonically
        with stake size (larger bet → stronger conviction → more belief-revealing text).
    """
    import json as _json

    df = _fix_bool_cols(df)
    results: dict = {"data_source": data_source}

    _log(f"\n{'='*72}")
    _log(f"[COMMITMENT VALIDITY] {data_source}")
    _log(f"{'='*72}")

    # ── Experiment 1: Wrong-Bettor Test ─────────────────────────────────────
    _log("\n  [Exp1] Wrong-Bettor Test")
    has_res = (
        "resolution" in df.columns
        and df["resolution"].astype(str).str.upper().isin(["YES", "NO"]).any()
    )
    if not has_res:
        _log("  [Exp1] resolution column unavailable (no resolved data yet), skipping")
        results["wrong_bettor_test"] = {"skip": "no_resolution_column"}
    elif "has_position" not in df.columns:
        _log("  [Exp1] has_position column missing, skipping")
        results["wrong_bettor_test"] = {"skip": "no_has_position_column"}
    else:
        hp = df[df["has_position"] == True].copy()
        hp["_res"]  = hp["resolution"].astype(str).str.strip().str.upper()
        hp["_gt"]   = hp["gt"].astype(str).str.strip().str.upper()
        hp["_pred"] = hp["pred"].astype(str).str.strip().str.upper()

        valid = hp[hp["_res"].isin(["YES", "NO"]) & hp["_gt"].isin(["YES", "NO"])]
        wrong = valid[valid["_gt"] != valid["_res"]]          # position_side ≠ resolution
        wrong_dir = wrong[wrong["_pred"].isin(["YES", "NO"])] # exclude NEUTRAL preds

        n_hp    = len(hp)
        n_wrong = len(wrong)
        n_dir   = len(wrong_dir)
        wrong_pct = n_wrong / max(n_hp, 1)

        _log(f"  Total HP comments: {n_hp}  |  wrong bettors: "
             f"{n_wrong} ({wrong_pct:.1%})")

        if n_dir < 20:
            _log(f"  [Exp1] Insufficient directional predictions n_dir={n_dir} < 20, skipping")
            results["wrong_bettor_test"] = {
                "n_hp": n_hp, "n_wrong_bettors": n_wrong,
                "n_directional": n_dir, "skip": f"n_dir={n_dir}<20",
            }
        else:
            # For wrong bettors: position_side ≠ resolution, so pred==gt and pred==resolution
            # are mutually exclusive for {YES,NO} preds — sum to n_dir.
            n_follows_commit = int(wrong_dir["correct"].sum())   # pred == position_side
            n_follows_truth  = n_dir - n_follows_commit          # pred == resolution
            pct_commit = n_follows_commit / n_dir

            p_commit = binom_pvalue_greater(n_follows_commit, n_dir, 0.5)
            sig = p_commit < 0.05

            _log(f"\n  ┌──────────────────────────────────────────────────────────")
            _log(f"  │  Wrong bettors (position_side!=resolution): {n_wrong}  "
                 f"directional preds: {n_dir}")
            _log(f"  │  following commitment(position_side): "
                 f"{n_follows_commit}/{n_dir} = {pct_commit:.3f}")
            _log(f"  │  following truth(resolution):          "
                 f"{n_follows_truth}/{n_dir} = {1-pct_commit:.3f}")
            _log(f"  │  Binomial p(follows_commitment > 0.5) = {p_commit:.4f}  "
                 f"{'[OK] commitment language shift significant ✓' if sig else '[!!] not significant'}")
            _log(f"  └──────────────────────────────────────────────────────────")

            results["wrong_bettor_test"] = {
                "n_hp": n_hp,
                "n_wrong_bettors": n_wrong,
                "wrong_bettor_rate": round(wrong_pct, 4),
                "n_directional": n_dir,
                "n_follows_commitment": n_follows_commit,
                "n_follows_truth": n_follows_truth,
                "pct_follows_commitment": round(pct_commit, 4),
                "binom_p": round(p_commit, 4),
                "significant": sig,
            }

    # ── Experiment 2: Dose-Response ──────────────────────────────────────────
    _log("\n  [Exp2] Position-Size Dose-Response")
    if "has_position" not in df.columns or "position_size" not in df.columns:
        _log("  [Exp2] has_position / position_size columns missing, skipping")
        results["dose_response"] = {"skip": "missing_columns"}
    else:
        hp_dir = df[
            (df["has_position"] == True) &
            (df["pred"].astype(str).str.upper().isin(["YES", "NO"])) &
            (df["position_size"].notna()) &
            (df["position_size"].astype(float) > 0)
        ].copy()
        hp_dir["position_size"] = hp_dir["position_size"].astype(float)

        if len(hp_dir) < 40:
            _log(f"  [Exp2] Insufficient HP directional samples n={len(hp_dir)} < 40, skipping")
            results["dose_response"] = {"skip": f"n_hp_dir={len(hp_dir)}<40"}
        else:
            qs = hp_dir["position_size"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values
            if qs[0] == qs[-1]:
                _log("  [Exp2] position_size has no variance, skipping")
                results["dose_response"] = {"skip": "no_spread_in_position_size"}
            else:
                q_labels = ["Q1(smallest)", "Q2", "Q3", "Q4(largest)"]
                hp_dir["_q"] = pd.cut(
                    hp_dir["position_size"],
                    bins=qs, labels=q_labels, include_lowest=True,
                )
                _log(f"\n  ┌──────────────────────────────────────────────────────────")
                _log(f"  │  {'group':<12} {'DA':>6}  {'n':>5}  {'med_pos':>12}  {'p':>8}  sig")
                _log(f"  │  {'-'*58}")

                dose_rows = []
                for lbl in q_labels:
                    grp = hp_dir[hp_dir["_q"] == lbl]
                    if len(grp) < 5:
                        continue
                    n   = len(grp)
                    k   = int(grp["correct"].sum())
                    da  = k / n
                    p   = binom_pvalue_greater(k, n, 0.5)
                    med = float(grp["position_size"].median())
                    sig_flag = "*" if p < 0.05 else ""
                    _log(f"  │  {lbl:<12} {da:>6.3f}  {n:>5}  "
                         f"{med:>12,.0f}  {p:>8.4f}  {sig_flag}")
                    dose_rows.append({
                        "label": lbl, "da": round(da, 4),
                        "n": n, "median_position_size": round(med, 2),
                        "binom_p": round(p, 4), "significant": p < 0.05,
                    })

                # Q4 vs Q1 monotonicity test
                q1_grp = hp_dir[hp_dir["_q"] == "Q1(smallest)"]
                q4_grp = hp_dir[hp_dir["_q"] == "Q4(largest)"]
                csd = float("nan"); mwu_p = 1.0; dose_sig = False
                if len(q1_grp) >= 10 and len(q4_grp) >= 10:
                    mwu_p  = mannwhitney_pvalue_greater(
                        q4_grp["correct"].astype(int).tolist(),
                        q1_grp["correct"].astype(int).tolist(),
                    )
                    csd      = q4_grp["correct"].mean() - q1_grp["correct"].mean()
                    dose_sig = mwu_p < 0.05
                    _log(f"  │  {'-'*58}")
                    _log(f"  │  CSD(Q4-Q1) = {csd:+.3f}  MWU p = {mwu_p:.4f}  "
                         f"{'[OK] dose-response significant ✓' if dose_sig else '[!!] not significant'}")
                _log(f"  └──────────────────────────────────────────────────────────")

                results["dose_response"] = {
                    "n_hp_directional": len(hp_dir),
                    "quartiles": dose_rows,
                    "csd_q4_q1": round(csd, 4) if not math.isnan(csd) else None,
                    "mwu_p_large_gt_small": round(mwu_p, 4),
                    "significant": dose_sig,
                }

    # ── Save JSON ────────────────────────────────────────────────────────────
    if out_subdir:
        fname = f"commitment_validity_{data_source.replace('/', '_')}.json"
        json_path = os.path.join(out_subdir, fname)
        try:
            with open(json_path, "w", encoding="utf-8") as fh:
                _json.dump(results, fh, indent=2, ensure_ascii=False)
            _log(f"\n  [COMMITMENT VALIDITY] Results saved: {json_path}")
        except Exception as exc:
            _log(f"  [WARN] Failed to save commitment validity JSON: {exc}", "warning")

    _log(f"{'='*72}\n")
    return results


# ── Within-User DiD (Experiment 3) ──────────────────────────────────────────
# Gold-standard validity check: compare the SAME user's comments in the SAME
# market before vs after they took a financial position.
# Controls for individual heterogeneity (same user) and market effects (same
# market). Only feasible on Manifold where bet timestamps and comment timestamps
# share the same user ID space (user_address == user_id in positions).
#
# Pre  group: comments before first bet → GT = resolution
# Post group: existing HP comments    → GT = position_side  (T1 CSV)
#
# Unified comparison: DA(post, GT=resolution) vs DA(pre, GT=resolution)
# Negative DiD → commitment biases language away from market truth.

def build_within_user_pre_comments(
    comments: pd.DataFrame,
    positions: pd.DataFrame,
    data_source: str,
) -> pd.DataFrame:
    """Return pre-commitment comments for within-user DiD.

    For each (user, market) where the user eventually took a YES/NO position,
    select their comments posted BEFORE the first bet timestamp.
    Sets _gt = resolution so the pre group can be tested with T1.

    Only implemented for Manifold (positions use user_id/contract_id matching
    comments' user_address/market_id). Returns empty DataFrame for Polymarket.
    """
    if positions is None or len(positions) == 0:
        return pd.DataFrame()

    pos_cols = {'user_id', 'contract_id', 'created_at', 'position_side'}
    if not pos_cols.issubset(set(positions.columns)):
        _log(f"  [DiD] {data_source}: positions missing required columns, skipping")
        return pd.DataFrame()

    # Restrict to qualified markets in this topic
    topic_mids = set(comments["market_id"].astype(str).unique())
    pos = positions[
        positions["contract_id"].astype(str).isin(topic_mids) &
        positions["position_side"].isin(["YES", "NO"])
    ].copy()
    if len(pos) == 0:
        return pd.DataFrame()

    pos["_ts"] = pd.to_datetime(pos["created_at"], utc=True, errors="coerce")
    pos = pos.dropna(subset=["_ts"])

    # First bet timestamp per (user, market)
    first_bet = (
        pos.groupby(["user_id", "contract_id"])["_ts"]
        .min()
        .reset_index()
        .rename(columns={"_ts": "first_bet_ts",
                         "user_id": "user_address",
                         "contract_id": "market_id"})
    )
    # Eventual position side (last bet in the market)
    eventual = (
        pos.sort_values("_ts")
        .groupby(["user_id", "contract_id"])["position_side"]
        .last()
        .reset_index()
        .rename(columns={"user_id": "user_address",
                         "contract_id": "market_id",
                         "position_side": "eventual_position_side"})
    )

    # Filter comments: quality + English + has resolution
    cmt = comments[
        comments["text"].apply(is_quality_text) &
        comments["text"].apply(is_english) &
        comments["resolution"].isin(["YES", "NO"])
    ].copy()
    cmt["_ts"] = pd.to_datetime(cmt["created_at"], utc=True, errors="coerce")
    cmt = cmt.dropna(subset=["_ts"])

    # Join
    merged = cmt.merge(first_bet, on=["user_address", "market_id"], how="inner")
    merged = merged.merge(eventual, on=["user_address", "market_id"], how="inner")
    merged["first_bet_ts"] = pd.to_datetime(merged["first_bet_ts"], utc=True, errors="coerce")

    # Pre-commitment: comment strictly before first bet
    pre = merged[merged["_ts"] < merged["first_bet_ts"]].copy()
    pre["_gt"]          = pre["resolution"]
    pre["has_position"] = False
    pre["position_size"] = 0.0

    n_pairs = pre.groupby(["user_address", "market_id"]).ngroups if len(pre) > 0 else 0
    _log(f"  [DiD] {data_source}: pre-commitment comments = {len(pre)}, "
         f"unique (user,market) pairs = {n_pairs}")
    return pre


def run_task1_did(
    pre_df: pd.DataFrame,
    data_source: str,
    out_subdir: str,
    n_each: int = 200,
) -> pd.DataFrame:
    """Run Task1 on pre-commitment comments (GT = resolution).

    Saves task1_did_pre_{ds}_{topic}.csv alongside the regular task1.csv.
    Then calls _report_task1_did to compare pre vs post.
    """
    if pre_df is None or len(pre_df) == 0:
        return pd.DataFrame()

    csv_path  = os.path.join(out_subdir,
                             f"qwen3_{data_source.replace('/','_')}_task1_did_pre.csv")
    state_key = f"{MODEL}/{data_source}/task1_did_pre"
    expected  = min(len(pre_df), n_each) if not RUN_FULL else len(pre_df)

    existing = _csv_load_partial(csv_path)
    results  = existing.to_dict("records") if len(existing) > 0 else []
    if len(existing) > 0 and "market_id" in existing.columns:
        done_keys = set(zip(existing["market_id"].astype(str),
                            existing["created_at"].astype(str)))
    else:
        done_keys = set()

    if _is_complete(csv_path, expected) or _state_is_done(state_key):
        _log(f"  [DiD] Task1-pre {data_source} already complete ({len(existing)} rows), skipping inference")
        _report_task1_did(existing, data_source, out_subdir)
        return existing

    sample = _sample_df(pre_df, None if RUN_FULL else expected)
    _topic_key = data_source.split("/")[-1] if "/" in data_source else ""

    _log(f"\n{'='*62}")
    _log(f"[{data_source.upper()}] DiD Task1-Pre -- pre-commitment comment inference "
         f"(GT=resolution, n={len(sample)})")
    _log(f"{'='*62}")

    def _did_pre_worker(row, _topic=_topic_key):
        _raw = query_llm(prompt_task1(row["question"], row["text"], topic=_topic))
        pred    = parse_yes_no(_raw)
        gt      = row["_gt"]   # = resolution
        correct = (pred == gt)
        print(f"  [DiD-pre] GT={gt:<3} PRED={pred:<7} {'[Y]' if correct else '[N]'}  "
              f"{str(row['text'])[:48]!r}")
        return {
            "market_id":              str(row.get("market_id", "")),
            "created_at":             str(row.get("created_at", "")),
            "user_address":           str(row.get("user_address", "") or ""),
            "question":               str(row.get("question", ""))[:200],
            "text":                   str(row["text"])[:120],
            "gt":                     gt,          # resolution
            "pred":                   pred,
            "correct":                correct,
            "has_position":           False,
            "position_size":          0.0,
            "resolution":             str(row.get("resolution", "") or "").strip().upper(),
            "eventual_position_side": str(row.get("eventual_position_side", "") or ""),
            "raw_response":           str(_raw)[:300],
            "model":                  MODEL,
        }

    sample_rows = [((str(r.get("market_id","")), str(r.get("created_at",""))), r)
                   for _, r in sample.iterrows()]
    results = _run_infer_loop(
        sample_rows, done_keys, _did_pre_worker, results, csv_path, expected
    )

    df_res = pd.DataFrame(results)
    df_res.to_csv(csv_path, index=False)
    _state_mark_done(state_key)
    _report_task1_did(df_res, data_source, out_subdir)
    return df_res


def _report_task1_did(
    pre_df: pd.DataFrame,
    data_source: str,
    out_subdir: str,
) -> dict:
    """Compare pre vs post T1 accuracy using resolution as unified GT.

    DiD = DA(post, GT=resolution) − DA(pre, GT=resolution).
    Negative DiD → post-commitment language drifts away from market truth,
    which is direct evidence of commitment-induced language bias.

    Also runs a per-(user,market)-pair sign test to assess individual-level
    statistical significance.
    """
    import json as _json

    pre_df  = _fix_bool_cols(pre_df)
    results = {"data_source": data_source}

    if pre_df is None or len(pre_df) == 0:
        results["skip"] = "empty_pre_df"
        return results

    # Load existing post T1 CSV
    post_csv = os.path.join(out_subdir,
                            f"qwen3_{data_source.replace('/','_')}_task1.csv")
    post_df  = _fix_bool_cols(_csv_load_partial(post_csv))

    if len(post_df) == 0:
        results["skip"] = "post_t1_csv_not_found"
        return results

    _log(f"\n{'='*72}")
    _log(f"[DiD] Within-User Pre vs Post — {data_source}")
    _log(f"{'='*72}")

    # ── Pre group (GT=resolution, already) ──────────────────────────────────
    pre_dir = pre_df[pre_df["pred"].astype(str).str.upper().isin(["YES","NO"])]
    n_pre   = len(pre_dir)
    da_pre  = pre_dir["correct"].mean() if n_pre > 0 else float("nan")

    # ── Post HP group: recompute accuracy against resolution ────────────────
    has_res_post = (
        "resolution" in post_df.columns and
        post_df["resolution"].astype(str).str.upper().isin(["YES","NO"]).any()
    )
    if has_res_post and "has_position" in post_df.columns:
        post_hp = post_df[post_df["has_position"] == True].copy()
        post_hp["_pred_u"] = post_hp["pred"].astype(str).str.upper()
        post_hp["_res_u"]  = post_hp["resolution"].astype(str).str.upper()
        post_hp_dir = post_hp[
            post_hp["_pred_u"].isin(["YES","NO"]) &
            post_hp["_res_u"].isin(["YES","NO"])
        ]
        n_post      = len(post_hp_dir)
        da_post_res = (post_hp_dir["_pred_u"] == post_hp_dir["_res_u"]).mean() if n_post > 0 else float("nan")
        da_post_pos = post_hp_dir["correct"].mean() if n_post > 0 else float("nan")
    else:
        n_post = 0; da_post_res = float("nan"); da_post_pos = float("nan")
        post_hp_dir = pd.DataFrame()

    did = (da_post_res - da_pre
           if not (math.isnan(da_pre) or math.isnan(da_post_res))
           else float("nan"))

    _log(f"  {'group':<30} {'n':>6}  {'DA(GT=resolution)':>18}  {'DA(GT=position)':>16}")
    _log(f"  {'-'*74}")
    _log(f"  {'Pre  (before-bet, GT=resolution)':<30} {n_pre:>6}  {da_pre:>18.3f}  {'--':>16}")
    _log(f"  {'Post HP (after-bet, GT=resolution)':<30} {n_post:>6}  {da_post_res:>18.3f}  {da_post_pos:>16.3f}")
    _log(f"  {'-'*74}")
    _log(f"  DiD = DA(post,res) - DA(pre,res) = {did:+.3f}  "
         f"{'-> language drifts from truth post-commitment (commitment bias ✓)' if (not math.isnan(did) and did < 0) else '-> language converges to truth post-commitment (information effect)' if (not math.isnan(did) and did > 0) else ''}")

    results.update({
        "n_pre_directional":       n_pre,
        "n_post_directional":      n_post,
        "da_pre_gt_resolution":    round(da_pre, 4) if not math.isnan(da_pre) else None,
        "da_post_gt_resolution":   round(da_post_res, 4) if not math.isnan(da_post_res) else None,
        "da_post_gt_position":     round(da_post_pos, 4) if not math.isnan(da_post_pos) else None,
        "did":                     round(did, 4) if not math.isnan(did) else None,
    })

    # ── Per-(user,market) pair sign test ─────────────────────────────────────
    uid_col = "user_address"
    if (uid_col in pre_dir.columns and "market_id" in pre_dir.columns and
            uid_col in post_hp_dir.columns and "market_id" in post_hp_dir.columns
            and len(post_hp_dir) > 0):

        pre_pairs = (
            pre_dir.groupby([uid_col, "market_id"])
            .apply(lambda g: (g["correct"]).mean())
            .reset_index(name="acc_pre")
        )
        post_hp_dir = post_hp_dir.copy()
        post_hp_dir["_correct_res"] = (post_hp_dir["_pred_u"] == post_hp_dir["_res_u"])
        post_pairs = (
            post_hp_dir.groupby([uid_col, "market_id"])
            .apply(lambda g: g["_correct_res"].mean())
            .reset_index(name="acc_post_res")
        )
        paired = pre_pairs.merge(post_pairs, on=[uid_col, "market_id"])
        n_pairs = len(paired)

        if n_pairs >= 10:
            diffs    = (paired["acc_post_res"] - paired["acc_pre"]).tolist()
            positive = sum(1 for d in diffs if d > 0)
            negative = sum(1 for d in diffs if d < 0)
            ties     = sum(1 for d in diffs if d == 0)
            n_no_tie = positive + negative
            mean_did = sum(diffs) / len(diffs)
            p_sign   = binom_pvalue_greater(positive, n_no_tie, 0.5) if n_no_tie > 0 else 1.0

            _log(f"\n  [Paired sign test]  {n_pairs} pairs (user,market)")
            _log(f"  DiD > 0 (post closer to truth): {positive}  "
                 f"DiD < 0 (post further from truth): {negative}  ties: {ties}")
            _log(f"  per-pair mean DiD = {mean_did:+.3f}  "
                 f"sign-test p = {p_sign:.4f}  "
                 f"{'[OK] significant ✓' if p_sign < 0.05 else '[!!] not significant'}")
            results["paired_sign_test"] = {
                "n_pairs": n_pairs,
                "n_positive_did": positive,
                "n_negative_did": negative,
                "n_ties": ties,
                "mean_did_per_pair": round(mean_did, 4),
                "sign_test_p": round(p_sign, 4),
                "significant": p_sign < 0.05,
                "direction": ("commitment_bias" if mean_did < 0
                              else "information_effect" if mean_did > 0
                              else "null"),
            }
        else:
            _log(f"  [Paired] only {n_pairs} pairs < 10, skipping paired test")
            results["paired_sign_test"] = {"skip": f"n_pairs={n_pairs}<10"}

    # ── Wrong-bettor subgroup ─────────────────────────────────────────────────
    if len(post_hp_dir) > 0 and "gt" in post_hp_dir.columns:
        wb = post_hp_dir[
            post_hp_dir["gt"].astype(str).str.upper() != post_hp_dir["_res_u"]
        ]
        if len(wb) >= 10:
            da_wb = (wb["_pred_u"] == wb["_res_u"]).mean()
            results["wrong_bettor_da_post_gt_resolution"] = round(float(da_wb), 4)
            _log(f"\n  Wrong bettors (post, GT=resolution): "
                 f"DA={da_wb:.3f}  n={len(wb)}  "
                 f"{'<- significantly below pre: evidence for commitment bias' if (not math.isnan(da_pre) and da_wb < da_pre) else ''}")

    _log(f"{'='*72}\n")

    # Save JSON
    if out_subdir:
        json_path = os.path.join(out_subdir,
                                 f"did_{data_source.replace('/','_')}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as fh:
                _json.dump(results, fh, indent=2, ensure_ascii=False)
            _log(f"  [DiD] Results saved: {json_path}")
        except Exception as exc:
            _log(f"  [DiD][WARN] Save failed: {exc}", "warning")

    return results


def run_task2(task2_pairs: pd.DataFrame, data_source: str, out_subdir: str,
              n_each: int = 80, variant: str = ""):
    """Run the main 4-class Task2 evaluation for one topic slice."""
    _log(f"\n{'='*62}")
    _sfx = f" [{variant}]" if variant else ""
    _log(f"[{data_source.upper()}] Phase1 Task2 — Language-Action Consistency{_sfx}")
    _log(f"{'='*62}")

    _fname_sfx = f"_{variant}" if variant else ""
    csv_path  = os.path.join(out_subdir, f"qwen3_{data_source.replace('/', '_')}_task2{_fname_sfx}.csv")
    state_key = f"{MODEL}/{data_source}/task2{_fname_sfx}"

    if task2_pairs.empty:
        _log("  [!!] No valid Task 2 pairs, skipping", "warning")
        return None

    samples = []
    for act in ["flip","increase","decrease","hold"]:
        sub = task2_pairs[task2_pairs["action"] == act]
        n   = len(sub) if RUN_FULL else min(n_each, len(sub))
        if n > 0:
            samples.append(_sample_df(sub, None if RUN_FULL else n))
            print(f"  {act:<10}: {n} samples (pool={len(sub)})")
    if not samples:
        _log("  [!!] Sampling failed", "warning")
        return None

    sample   = pd.concat(samples).sample(frac=1, random_state=SEED).reset_index(drop=True)
    expected = len(sample)

    existing = _csv_load_partial(csv_path)
    if len(existing) > 0 and "market_id" in existing.columns and "created_at" in existing.columns:
        done_keys = set(zip(existing["market_id"].astype(str), existing["created_at"].astype(str)))
    else:
        done_keys = set()
    results  = existing.to_dict("records") if len(existing) > 0 else []

    if _is_complete(csv_path, expected) or _state_is_done(state_key):
        _log(f"  [RESUME] Task2 {data_source} already complete ({len(existing)} rows), skipping LLM calls")
        logger.info(f"SKIP task2 {data_source} — {len(existing)} rows exist")
        _report_task2(existing, data_source)
        return existing

    if done_keys:
        _log(f"  [RESUME] Task2 {data_source} partial progress ({len(done_keys)}/{expected}), continuing")
        logger.info(f"RESUME task2 {data_source}: {len(done_keys)} rows done")

    if RUN_FULL:
        _log(f"Full run: {expected} Task2 rows ({len(done_keys)} already done)")
    else:
        _log(f"Sampling: {expected} rows (already done: {len(done_keys)} rows)")

    def _t2_worker(row):
        pred_raw = query_llm(prompt_task2(row["question"], row["text"],
                                          row["position_side"], row["position_size"]))
        pred    = parse_action(pred_raw)
        gt      = row["action"]
        correct = (pred == gt)
        print(f"  GT={gt:<10} PRED={pred:<10} {'[Y]' if correct else '[N]'}  "
              f"${row['position_size']:,.0f}  {str(row['text'])[:42]!r}")
        return {"market_id": row.get("market_id", ""), "created_at": row.get("created_at", ""),
                "question": str(row.get("question", ""))[:200],
                "text": str(row["text"])[:120],
                "gt": gt, "pred": pred, "correct": correct,
                "has_position": True,
                "position_size": row["position_size"],
                "raw_response": str(pred_raw)[:300],
                "model": MODEL}

    sample_rows = [((str(r.get("market_id", "")), str(r.get("created_at", ""))), r)
                   for _, r in sample.iterrows()]
    results = _run_infer_loop(sample_rows, done_keys, _t2_worker, results, csv_path, expected)

    df_res = pd.DataFrame(results)
    df_res.to_csv(csv_path, index=False)
    _state_mark_done(state_key)
    _report_task2(df_res, data_source)
    return df_res


def _report_task2(df: pd.DataFrame, data_source: str):
    """Report Task2 using both 4-class accuracy and active-action accuracy."""
    total   = len(df)
    if total == 0 or "correct" not in df.columns:
        print(f"\n------ [{data_source.upper()}] Task2 -- No data ------")
        return
    df = _fix_bool_cols(df)
    correct = int(df["correct"].sum())

    acc_4c = correct / total
    p_4c   = binom_pvalue_greater(correct, total, p=0.25)

    df_act_pred = df[df["pred"] != "hold"]
    acc_act_pred = df_act_pred["correct"].mean() if len(df_act_pred) > 0 else float("nan")
    p_act_pred   = binom_pvalue_greater(int(df_act_pred["correct"].sum()), len(df_act_pred), p=0.333) \
                   if len(df_act_pred) > 1 else 1.0  # three-class baseline:1/3

    df_act_gt   = df[df["gt"] != "hold"]
    acc_act_gt  = df_act_gt["correct"].mean() if len(df_act_gt) > 0 else float("nan")
    p_act_gt    = binom_pvalue_greater(int(df_act_gt["correct"].sum()), len(df_act_gt), p=0.333) \
                  if len(df_act_gt) > 1 else 1.0

    hold_rate_pred = (df["pred"] == "hold").mean()
    hold_rate_gt   = (df["gt"]   == "hold").mean()

    print(f"\n------ [{data_source.upper()}] Task2 ------")
    print(f"  Total samples:{total}  Prediction distribution: {dict(Counter(df['pred']))}")
    print(f"  [Note] All Task2 samples have a held position (has_position=True); no no-position group (see Task1 NP group)")
    print(f"  hold rate: GT={hold_rate_gt:.1%}  LLM_pred={hold_rate_pred:.1%}  (hold-bias indicator)")
    print(f"  [4c]        acc={acc_4c:.3f}   p(>0.25)={p_4c:.4f}  "
          f"{'[OK]' if p_4c<0.05 else '[!!]'}  (4-class, all categories, current threshold)")
    print(f"  [active-pred] acc={acc_act_pred:.3f}  p(>0.33)={p_act_pred:.4f}  n={len(df_act_pred)}  "
          f"{'[OK]' if p_act_pred<0.05 else '[!!]'}  (precision when the LLM predicts an active move)")
    print(f"  [active-gt]  acc={acc_act_gt:.3f}  p(>0.33)={p_act_gt:.4f}  n={len(df_act_gt)}  "
          f"{'[OK]' if p_act_gt<0.05 else '[!!]'}  (recall when GT is an active action)")
    for act in ["flip","increase","decrease","hold"]:
        sub_gt   = df[df["gt"]   == act]
        sub_pred = df[df["pred"] == act]
        if len(sub_gt):
            print(f"  Recall({act:<10}): {sub_gt['correct'].mean():.3f}  n_gt={len(sub_gt)}  "
                  f"n_pred={len(sub_pred)}")


def run_task2_binary(task2_pairs: pd.DataFrame, data_source: str, out_subdir: str,
                     n_each: int = 80, model: str = None):
    """Run the two Task2 binary subtasks: flip-vs-hold and decrease-vs-hold."""
    _log(f"\n{'='*62}")
    _log(f"[{data_source.upper()}] Phase1 Task2-Binary — flip-vs-hold & decrease-vs-hold")
    _log(f"{'='*62}")

    if task2_pairs.empty:
        _log("  [!!] No valid Task2 pairs, skipping T2-binary", "warning")
        return {"flip_hold": None, "decrease_hold": None}

    results = {}

    for subtask, actions, prompt_fn, label, p_base in [
        ("flip_hold",     ["flip",     "hold"], prompt_task2_flip_hold,     "flip-vs-hold",     0.5),
        ("decrease_hold", ["decrease", "hold"], prompt_task2_decrease_hold, "decrease-vs-hold", 0.5),
    ]:
        csv_path  = os.path.join(out_subdir,
                                 f"qwen3_{data_source.replace('/', '_')}_task2_{subtask}.csv")
        state_key = f"{MODEL}/{data_source}/task2_{subtask}"

        # Sample each label bucket independently for a stable binary Task2 subset.
        samples = []
        sample_sizes = {}
        for act in actions:
            sub = task2_pairs[task2_pairs["action"] == act]
            pool_n = len(sub)
            n = pool_n if RUN_FULL else min(n_each, pool_n)
            sample_sizes[act] = {"pool": pool_n, "sampled": n}
            if n > 0:
                samples.append(_sample_df(sub, None if RUN_FULL else n))
                _log(f"  [{subtask}] {act:<10}: sampled={n} (pool={pool_n})"
                     f"{'  [!!] small pool; result may be unstable' if pool_n < 10 else ''}")
            else:
                _log(f"  [{subtask}] {act:<10}: pool=0, skipping")
        if not samples:
            _log(f"  [!!] [{subtask}] sampling failed, skipping", "warning")
            results[subtask] = None
            continue

        sample   = pd.concat(samples).sample(frac=1, random_state=SEED).reset_index(drop=True)
        expected = len(sample)

        existing = _csv_load_partial(csv_path)
        if len(existing) > 0 and "market_id" in existing.columns and "created_at" in existing.columns:
            done_keys = set(zip(existing["market_id"].astype(str), existing["created_at"].astype(str)))
        else:
            done_keys = set()
        rows     = existing.to_dict("records") if len(existing) > 0 else []

        if _is_complete(csv_path, expected) or _state_is_done(state_key):
            _log(f"  [RESUME] Task2-{subtask} {data_source} already complete ({len(existing)} rows), skipping")
            results[subtask] = existing
            continue

        if done_keys:
            _log(f"  [RESUME] Task2-{subtask} {data_source} partial progress ({len(done_keys)}/{expected}), continuing")

        _log(f"  [{label}] Start inference ({len(done_keys)}/{expected} already done)")

        def _t2b_worker(row, _pf=prompt_fn, _acts=actions, _st=subtask, _m=model):
            pred_raw = query_llm(_pf(row["question"], row["text"],
                                     row["position_side"], row["position_size"]), model=_m)
            pred = pred_raw.strip().lower().split()[0] if pred_raw.strip() else "hold"
            if pred not in set(_acts):
                pred = "hold"
            gt      = row["action"]
            correct = (pred == gt)
            print(f"  [{_st}] GT={gt:<10} PRED={pred:<10} "
                  f"{'[Y]' if correct else '[N]'}  {str(row['text'])[:42]!r}")
            return {"market_id": row.get("market_id", ""), "created_at": row.get("created_at", ""),
                    "text": str(row["text"])[:120],
                    "gt": gt, "pred": pred, "correct": correct}

        sample_rows = [((str(r.get("market_id", "")), str(r.get("created_at", ""))), r)
                       for _, r in sample.iterrows()]
        rows = _run_infer_loop(sample_rows, done_keys, _t2b_worker, rows, csv_path, expected)

        df_res = pd.DataFrame(rows)
        df_res.to_csv(csv_path, index=False)
        _state_mark_done(state_key)
        results[subtask] = df_res

    _report_task2_binary(results, data_source)
    return results


def _report_task2_binary(results_dict: dict, data_source: str):
    """Report the two Task2 binary subtasks."""
    for subtask, label, p_base in [
        ("flip_hold",     "flip-vs-hold",     0.5),
        ("decrease_hold", "decrease-vs-hold", 0.5),
    ]:
        df = results_dict.get(subtask)
        if df is None or len(df) == 0:
            print(f"  [{data_source.upper()}] Task2-{label}: no data")
            continue
        df = _fix_bool_cols(df)
        total   = len(df)
        correct = int(df["correct"].sum())
        acc     = correct / total
        p_val   = binom_pvalue_greater(correct, total, p=p_base)
        for act in df["gt"].unique():
            sub_gt = df[df["gt"] == act]
            recall = sub_gt["correct"].mean() if len(sub_gt) > 0 else float("nan")
            print(f"    Recall({act:<10}): {recall:.3f}  n={len(sub_gt)}")
        print(f"  [{data_source.upper()}] Task2-{label}: "
              f"acc={acc:.3f}  p(>0.5)={p_val:.4f}  "
              f"{'[OK]' if p_val < 0.05 else '[!!]'}  n={total}")




def run_task2_multi_threshold(comments_q: pd.DataFrame, positions: pd.DataFrame,
                               data_source: str, out_subdir: str,
                               n_each: int = 80, model: str = None) -> dict:
    """
    [v11-5] Build Task2 pairs independently for each hold-hour threshold in TASK2_HOLD_HOURS_LIST
    and evaluate to compare how observation window length affects language-action consistency.
    Returns {hours: df_result} (48h is also included here for comparison with the main pipeline).
    Returns {hours: df_result} (48h is also included here for comparison with the main pipeline).
    """
    _log(f"\n{'='*62}")
    _log(f"[{data_source.upper()}] Phase1 Task2 -- Multi-threshold: {TASK2_HOLD_HOURS_LIST}h")
    _log(f"{'='*62}")

    _ds_platform = data_source.split("/")[0]

    results = {}
    for h in TASK2_HOLD_HOURS_LIST:
        _log(f"\n  [T2-MultiThreshold] hold_hours={h}h -- Building pairs ...")
        pairs = build_task2_pairs(comments_q, positions, _ds_platform, hold_hours=h)
        _log(f"  [T2-MultiThreshold] hold_hours={h}h — pairs={len(pairs)}")

        if pairs.empty:
            _log(f"  [!!] hold_hours={h}h: no valid pairs, skipping", "warning")
            results[h] = None
            continue

        df = run_task2(pairs, data_source, out_subdir,
                       n_each=n_each, variant=f"h{h}")
        results[h] = df

    _report_task2_multi_threshold(results, data_source)
    return results


def _report_task2_multi_threshold(results: dict, data_source: str):
    """Summarize Task2 metrics across multiple hold-hour thresholds."""
    _log(f"\n------ [{data_source.upper()}] Task2 Multi-threshold comparison ------")
    _log(f"  {'hours':>8}  {'n':>6}  {'acc_4c':>8}  {'p_4c':>8}  {'hold_rate_pred':>14}  {'hold_rate_gt':>12}")
    for h in TASK2_HOLD_HOURS_LIST:
        df = results.get(h)
        if df is None or len(df) == 0:
            _log(f"  {h:>7}h   {'N/A':>6}  {'N/A':>8}  {'N/A':>8}  {'N/A':>14}  {'N/A':>12}")
            continue
        df = _fix_bool_cols(df)
        total = len(df)
        acc = df["correct"].mean()
        p_val = binom_pvalue_greater(int(df["correct"].sum()), total, p=0.25)
        hold_pred = (df["pred"] == "hold").mean()
        hold_gt = (df["gt"] == "hold").mean()
        sig = "[OK]" if p_val < 0.05 else "[!!]"
        _log(f"  {h:>7}h  {total:>6}  {acc:>8.3f}  {p_val:>8.4f}{sig:4}  {hold_pred:>14.1%}  {hold_gt:>12.1%}")


def run_task3(all_windows: list, data_source: str, out_subdir: str, n_win: int = 50,
              variant: str = "time"):
    """Run Task3 on either time-based or comment-count-based windows."""
    _log(f"\n{'='*62}")
    _log(f"[{data_source.upper()}] Phase1 Task3' — Belief-Price Lead [{variant} window]")
    _log(f"{'='*62}")

    csv_path  = os.path.join(out_subdir, f"qwen3_{data_source.replace('/', '_')}_task3_{variant}.csv")
    state_key = f"{MODEL}/{data_source}/task3_{variant}"

    # V15: diagnostic breakdown before applying price filter
    n_total = len(all_windows)
    n_consensus_ok = sum(1 for w in all_windows if w["comment_consensus"] is not None)
    n_next_odds_ok  = sum(1 for w in all_windows if w["next_window_avg_odds"] is not None)
    price_diffs = [
        abs(w["next_window_avg_odds"] - w["avg_odds"])
        for w in all_windows
        if w["next_window_avg_odds"] is not None
    ]
    _log(
        f"  [V16-T3] window screening: total={n_total} | consensus_valid={n_consensus_ok} | "
        f"next_odds_valid={n_next_odds_ok} | price_eps_threshold={TASK3_PRICE_EPS}"
    )
    if price_diffs:
        sorted_diffs = sorted(price_diffs)
        _log(
            f"  [V15-T3] price_diff stats (n={len(price_diffs)}): "
            f"min={sorted_diffs[0]:.4f} | median={sorted_diffs[len(sorted_diffs)//2]:.4f} | "
            f"max={sorted_diffs[-1]:.4f} | "
            f"above_threshold={sum(1 for d in price_diffs if d > TASK3_PRICE_EPS)}"
        )

    wins = [w for w in all_windows
            if w["comment_consensus"] is not None
            and w["next_window_avg_odds"] is not None
            and abs(w["next_window_avg_odds"] - w["avg_odds"]) > TASK3_PRICE_EPS]

    if not wins:
        _log(
            f"[V18][!!] EARLY-STOP (price_filter): 0 valid windows for {data_source}/{variant}. "
            f"All {n_total} windows failed at least one of: consensus_valid ({n_consensus_ok}/{n_total}), "
            f"next_odds_valid ({n_next_odds_ok}/{n_total}), or price_diff > {TASK3_PRICE_EPS} "
            f"({sum(1 for d in price_diffs if d > TASK3_PRICE_EPS)}/{len(price_diffs)}). "
            f"Returning None — this topic will be skipped.",
            "warning",
        )
        return None

    _log(
        f"  [V15-T3] gating config: min_comments_per_window={TASK3_MIN_WINDOW_COMMENTS} | "
        f"price_eps={TASK3_PRICE_EPS} | near50=[{TASK3_NEAR50_LOWER:.2f},{TASK3_NEAR50_UPPER:.2f}] | "
        f"wins_after_filter={len(wins)}"
    )
    _rng = random.Random(SEED)
    _rng.shuffle(wins)
    if RUN_FULL:
        near50 = [w for w in wins if TASK3_NEAR50_LOWER <= w["avg_odds"] <= TASK3_NEAR50_UPPER]
        n_near = len(near50)
        sample = list(wins)
        _rng.shuffle(sample)
        far50 = [w for w in wins if w not in near50]
    else:
        near50 = [w for w in wins if TASK3_NEAR50_LOWER <= w["avg_odds"] <= TASK3_NEAR50_UPPER]
        far50  = [w for w in wins if w not in near50]
        n_near = min(int(n_win * 0.6), len(near50))
        n_far  = min(n_win - n_near, len(far50))
        sample = near50[:n_near] + far50[:n_far]
        if len(sample) < n_win:
            rest = [w for w in wins if w not in sample]
            sample += rest[:n_win - len(sample)]
        _rng.shuffle(sample)

    # V15: log sampling outcome to diagnose near/far50 shortfalls
    _log(
        f"  [V15-T3] sampling: wins={len(wins)} | near50={len(near50)} | far50={len(far50)} | "
        f"requested_n_win={n_win} | actual_sample={len(sample)} | RUN_FULL={RUN_FULL}"
    )
    if len(sample) < n_win:
        _log(
            f"  [V15-T3][WARN] SAMPLE SHORTFALL: got {len(sample)} of requested {n_win} windows "
            f"(near50={len(near50)}, far50={len(far50)}). "
            f"near50 range: [{TASK3_NEAR50_LOWER},{TASK3_NEAR50_UPPER}]. "
            f"This topic will run with fewer windows than configured.",
            "warning",
        )

    existing   = _csv_load_partial(csv_path)
    done_keys  = set()
    records    = []
    if len(existing) > 0:
        # Rebuild stable window keys from existing CSV rows.
        for _, row in existing.iterrows():
            done_keys.add((round(float(row["avg_odds"]), 6),
                           round(float(row["next_odds"]), 6)))
        records = existing.to_dict("records")

    if _is_complete(csv_path, len(sample)) or _state_is_done(state_key):
        completion_ratio = len(existing) / len(sample) if len(sample) > 0 else 1.0
        _log(
            f"  [V18][RESUME] EARLY-STOP (csv_complete): skipping LLM calls for {data_source}/{variant}. "
            f"existing={len(existing)} / expected={len(sample)} rows ({completion_ratio:.0%}). "
            f"csv={csv_path}",
        )
        logger.info(f"SKIP task3 {data_source}/{variant} — {len(existing)}/{len(sample)} rows ({completion_ratio:.0%})")
        _report_task3(existing, data_source)
        return existing

    if done_keys:
        _log(f"  [RESUME] Task3 {data_source} partial progress ({len(done_keys)}/{len(sample)}), continuing")
        logger.info(f"RESUME task3 {data_source}: {len(done_keys)} windows already done")

    if RUN_FULL:
        _log(f"Full run: {len(sample)} windows selected (near-50={n_near}, done={len(done_keys)})\n")
    else:
        _log(f"Sample {len(sample)} windows (near-50: {n_near}, already done: {len(done_keys)})\n")

    def _t3_worker(win):
        clines  = "\n".join(f"  - {str(c['text'])[:120]}" for c in win["comments"][:10])
        _raw_t3 = query_llm(prompt_task3_lead(win["question"], clines, win["avg_odds"]))
        pred    = parse_up_down(_raw_t3)
        delta   = win["next_window_avg_odds"] - win["avg_odds"]
        gt_dir  = "UP" if delta > 0 else "DOWN"
        strength      = win.get("consensus_strength", 1.0)
        consensus     = win.get("comment_consensus")
        consensus_dir = win.get("comment_consensus_dir")
        stat_3c = "UP" if consensus == "YES" else "DOWN"
        stat_2c = "UP" if consensus_dir == "YES" else ("DOWN" if consensus_dir == "NO" else "NEUTRAL")
        correct_llm_3c  = (pred == gt_dir)
        correct_stat_3c = (stat_3c == gt_dir)
        correct_stat_2c = (stat_2c != "NEUTRAL") and (stat_2c == gt_dir)
        print(f"  odds={win['avg_odds']:.2f} str={strength:.2f} Δ={delta:+.3f} "
              f"GT={gt_dir:<4} s3c={stat_3c:<4}[{'Y' if correct_stat_3c else 'N'}] "
              f"s2c={stat_2c:<7}[{'Y' if correct_stat_2c else 'N'}] "
              f"llm={pred:<7}[{'Y' if correct_llm_3c else 'N'}]")
        return {
            "window_id":       win.get("_win_idx", 0),
            "avg_odds":        win["avg_odds"],
            "next_odds":       win["next_window_avg_odds"],
            "delta":           delta,
            "gt_dir":          gt_dir,
            "stat_strength":   strength,
            "stat_3c":         stat_3c,
            "stat_2c":         stat_2c,
            "stat_correct_3c": correct_stat_3c,
            "stat_correct_2c": correct_stat_2c,
            "llm_pred":        pred,
            "llm_correct":     correct_llm_3c,
            "naive_correct":   (gt_dir == "DOWN"),
            "pos_frac":        win.get("pos_comment_frac", float("nan")),
            "resolution":      win["resolution"],
            "question":        str(win.get("question", ""))[:200],
            "n_comments":      len(win.get("comments", [])),
            "raw_response":    str(_raw_t3)[:300],
            "model":           MODEL,
        }

    sample_rows = [
        ((round(win["avg_odds"], 6), round(win["next_window_avg_odds"], 6)),
         {**win, "_win_idx": wi})
        for wi, win in enumerate(sample)
    ]
    records = _run_infer_loop(
        sample_rows, done_keys, _t3_worker, records, csv_path, len(sample),
        checkpoint_every=10,
    )

    df_t3 = pd.DataFrame(records)
    df_t3.to_csv(csv_path, index=False)
    _state_mark_done(state_key)
    _report_task3(df_t3, data_source)
    return df_t3


def _report_task3(df: pd.DataFrame, data_source: str):
    total = len(df)
    if total == 0:
        print(f"\n------ [{data_source.upper()}] Task3' -- No data ------")
        return
    df = _fix_bool_cols(df)

    llm_dir = df[df["llm_pred"] != "NEUTRAL"]
    llm_cov = len(llm_dir) / total

    llm_3c_n   = total
    llm_3c_cor = int(df["llm_correct"].sum())
    bpla_llm_3c = llm_3c_cor / llm_3c_n
    p_llm_3c    = binom_pvalue_greater(llm_3c_cor, llm_3c_n)

    llm_2c_n   = len(llm_dir)
    llm_2c_cor = int(llm_dir["llm_correct"].sum())
    bpla_llm_2c = llm_2c_cor / llm_2c_n if llm_2c_n > 0 else float("nan")
    p_llm_2c    = binom_pvalue_greater(llm_2c_cor, llm_2c_n) if llm_2c_n > 1 else 1.0

    stat_3c_col  = "stat_correct_3c" if "stat_correct_3c" in df.columns else "stat_correct"
    stat_3c_cor  = int(df[stat_3c_col].sum())
    bpla_stat_3c = stat_3c_cor / total
    p_stat_3c    = binom_pvalue_greater(stat_3c_cor, total)

    # 2c: evaluate only strong-consensus windows when those columns exist.
    if "stat_2c" in df.columns and "stat_correct_2c" in df.columns:
        stat_dir_df  = df[df["stat_2c"] != "NEUTRAL"]
        stat_2c_n    = len(stat_dir_df)
        stat_2c_cor  = int(stat_dir_df["stat_correct_2c"].sum())
    else:
        # Backward compatibility for older CSVs without explicit 2c columns.
        stat_dir_df  = df
        stat_2c_n    = total
        stat_2c_cor  = stat_3c_cor
    bpla_stat_2c = stat_2c_cor / stat_2c_n if stat_2c_n > 0 else float("nan")
    p_stat_2c    = binom_pvalue_greater(stat_2c_cor, stat_2c_n) if stat_2c_n > 1 else 1.0
    stat_2c_cov  = stat_2c_n / total

    gt_dist   = dict(Counter(df["gt_dir"]))
    n_down_gt = gt_dist.get("DOWN", 0)
    n_up_gt   = gt_dist.get("UP",   0)
    bpla_naive       = n_down_gt / total  # Naive baseline: always predict DOWN
    bpla_above_naive = bpla_stat_3c - bpla_naive

    df_up   = df[df["gt_dir"] == "UP"]
    df_down = df[df["gt_dir"] == "DOWN"]
    bpla_stat_up   = df_up[stat_3c_col].mean()   if len(df_up)   > 0 else float("nan")
    bpla_stat_down = df_down[stat_3c_col].mean() if len(df_down) > 0 else float("nan")
    bpla_llm_up    = df_up["llm_correct"].mean()   if len(df_up)   > 0 else float("nan")
    bpla_llm_down  = df_down["llm_correct"].mean() if len(df_down) > 0 else float("nan")

    print(f"\n------ [{data_source.upper()}] Task3' ------")
    print(f"  Total windows: {total}  UP/DOWN: {gt_dist}  "
          f"naive_DOWNbaseline={bpla_naive:.3f}")
    print(f"  |-------- Stat ------------------------------------------------------------------------------------------------------------------------------------------")
    print(f"  └ [3c] BPLA={bpla_stat_3c:.3f}  p={p_stat_3c:.4f}  n={total}  "
          f"above_naive={bpla_above_naive:+.3f}  {'[OK]' if p_stat_3c<0.10 else '[!!]'}")
    print(f"  └ [2c] BPLA={bpla_stat_2c:.3f}  p={p_stat_2c:.4f}  n={stat_2c_n}  "
          f"coverage={stat_2c_cov:.1%}  {'[OK]' if p_stat_2c<0.10 else '[!!]'}")
    print(f"  |-- [Stratified] UP_acc={bpla_stat_up:.3f}(n={n_up_gt})  DOWN_acc={bpla_stat_down:.3f}(n={n_down_gt})")
    print(f"  |-------- LLM ----------------------------------------------------------------------------------------------------------------------------------------------")
    print(f"     [3c] BPLA={bpla_llm_3c:.3f}  p={p_llm_3c:.4f}  n={llm_3c_n}  "
          f"{'[OK]' if p_llm_3c<0.10 else '[!!]'}")
    print(f"     [2c] BPLA={bpla_llm_2c:.3f}  p={p_llm_2c:.4f}  n={llm_2c_n}  "
          f"coverage={llm_cov:.1%}  {'[OK]' if p_llm_2c<0.10 else '[!!]'}")
    print(f"     [stratified] UP_acc={bpla_llm_up:.3f}(n={n_up_gt})  DOWN_acc={bpla_llm_down:.3f}(n={n_down_gt})")

    # Compare high vs low position-comment fraction windows when available.
    if "pos_frac" in df.columns and df["pos_frac"].notna().any():
        POS_FRAC_THR = 0.5
        df_hp = df[df["pos_frac"] >= POS_FRAC_THR]
        df_lp = df[df["pos_frac"] <  POS_FRAC_THR]
        def _grp_bpla(gdf):
            if len(gdf) == 0: return float("nan"), float("nan"), float("nan")
            s3 = gdf[stat_3c_col].mean()
            l3 = gdf["llm_correct"].mean()
            naive = (gdf["gt_dir"] == "DOWN").mean()
            return s3, l3, naive
        s_hp, l_hp, naive_hp = _grp_bpla(df_hp)
        s_lp, l_lp, naive_lp = _grp_bpla(df_lp)
        print("  --- Position-fraction group comparison for Task3 BPLA ---")
        print(f"  [high pos_frac >= {POS_FRAC_THR:.0%}] n={len(df_hp):3d}  "
              f"stat_3c={s_hp:.3f}  llm_3c={l_hp:.3f}  naive={naive_hp:.3f}  "
              f"above_naive(stat)={s_hp-naive_hp:+.3f}")
        print(f"  [low  pos_frac <  {POS_FRAC_THR:.0%}] n={len(df_lp):3d}  "
              f"stat_3c={s_lp:.3f}  llm_3c={l_lp:.3f}  naive={naive_lp:.3f}  "
              f"above_naive(stat)={s_lp-naive_lp:+.3f}")
        print(f"  |--------------------------------------------------------------------------------------------------------------------------------------------------------------------")



def _report_funnel_summary(data_source: str, n_t0_pool: int, n_t1_hp: int,
                           n_t2_pairs: int, n_t3_windows: int):
    """Print sample counts across the benchmark funnel."""
    print(f"\n  {'-' * 16} [{data_source.upper()}] Funnel Summary {'-' * 16}")
    print(f"  T0 explicit_pos_pool : {n_t0_pool:>6}")
    print(f"  T1 HP samples        : {n_t1_hp:>6}")
    print(f"  T2 pairs             : {n_t2_pairs:>6}")
    print(f"  T3 eligible windows  : {n_t3_windows:>6}")
    ratios = []
    for prev, curr in [(n_t0_pool, n_t1_hp), (n_t1_hp, n_t2_pairs), (n_t2_pairs, n_t3_windows)]:
        ratios.append(f"{curr/max(prev,1):.1%}" if prev > 0 else "N/A")
    print(f"  Conversion rates      : T0->T1={ratios[0]}  T1->T2={ratios[1]}  T2->T3={ratios[2]}")
    print(f"  {'-' * 52}")


def _report_benchmark_progression(all_summary: dict, model_label: str = ""):
    """Print per-topic T0-T3 progression using compact significance labels."""
    SIG = 0.05

    def _sig_t3(v_acc, v_p, above_naive):
        """Task3 counts as strong only when significant and above the naive baseline."""
        if isinstance(v_acc, float) and math.isnan(v_acc):
            return "NA"
        if isinstance(v_p, float) and math.isnan(v_p):
            return "NA"
        if v_p >= SIG:
            return "NS"
        if isinstance(above_naive, float) and math.isnan(above_naive):
            return "SIG"
        return "SIG" if above_naive > 0 else "WEAK"

    def _sig(v_acc, v_p, baseline):
        if isinstance(v_acc, float) and math.isnan(v_acc):
            return "NA"
        if isinstance(v_p, float) and math.isnan(v_p):
            return "NA"
        return "SIG" if v_p < SIG else "NS"

    model_suffix = f" [{model_label}]" if model_label else ""
    print(f"\n\n{'=' * 86}")
    print(f"  CommitBench progression (T0->T1->T2->T3){model_suffix}")
    print("  SIG=strong signal  WEAK=significant but not above naive  NS=not significant  NA=no sample")
    print("  Random baselines: T0=0.5, T1=0.5, T2=0.25, T3=0.5")
    print(f"{'=' * 86}")

    pm_keys  = sorted(k for k in all_summary if k[0] == "polymarket")
    man_keys = sorted(k for k in all_summary if k[0] == "manifold")
    all_keys = pm_keys + man_keys

    hdr = (f"  {'topic':<28}  {'T0':^6}  {'T1(DA)':^8}  {'T2':^6}  {'T3(stat)':^10}"
           f"  {'T2b-flip':^8}  {'T2b-dec':^8}  {'first_fail':^10}  {'score':^6}")
    print(hdr)
    print("  " + "-" * 84)

    for ds, topic in all_keys:
        s = all_summary[(ds, topic)]
        t0 = _sig(s.get("t0_acc"),     s.get("t0_p"),     0.5)
        t1 = _sig(s.get("t1_acc_2c"),  s.get("t1_p_da"),  0.5)
        t2 = _sig(s.get("t2_acc_4c"),  s.get("t2_p_4c"),  0.25)
        t3 = _sig_t3(s.get("t3_stat_3c"), s.get("t3_p_stat"), s.get("t3_above_naive", float("nan")))
        t2b_flip = _sig(s.get("t2b_flip_acc"), s.get("t2b_flip_p"), 0.5)
        t2b_dec  = _sig(s.get("t2b_dec_acc"),  s.get("t2b_dec_p"),  0.5)
        stages = [("T0", t0), ("T1", t1), ("T2", t2), ("T3", t3)]
        first_fail = next((name for name, v in stages if v in ("NS", "WEAK")), "none")
        if all(v == "NA" for _, v in stages):
            first_fail = "N/A"
        score = sum(1 for _, v in stages if v == "SIG")
        total_valid = sum(1 for _, v in stages if v != "NA")
        score_str = f"{score}/{total_valid}" if total_valid > 0 else "NA"

        label = f"{ds}/{topic}"
        print(f"  {label:<28}  {t0:^6}  {t1:^8}  {t2:^6}  {t3:^10}"
              f"  {t2b_flip:^8}  {t2b_dec:^8}  {first_fail:^8}  {score_str:^6}")

    print(f"{'=' * 86}")
    all_t0 = [all_summary[k].get("t0_p", 1.0) for k in all_keys]
    all_t1 = [all_summary[k].get("t1_p_da", 1.0) for k in all_keys]
    all_t2 = [all_summary[k].get("t2_p_4c", 1.0) for k in all_keys]
    all_t3_real = []
    for k in all_keys:
        s = all_summary[k]
        p = s.get("t3_p_stat", 1.0)
        ab = s.get("t3_above_naive", float("nan"))
        is_real = (not (isinstance(p, float) and math.isnan(p))
                   and p < SIG
                   and not (isinstance(ab, float) and math.isnan(ab))
                   and ab > 0)
        all_t3_real.append(is_real)

    def _cnt(ps):
        return sum(1 for p in ps if not (isinstance(p, float) and math.isnan(p)) and p < SIG)
    n = len(all_keys)
    print(f"  Significant topics: T0={_cnt(all_t0)}/{n}  T1={_cnt(all_t1)}/{n}  "
          f"T2={_cnt(all_t2)}/{n}  T3_real={sum(all_t3_real)}/{n}")
    print(f"{'=' * 86}\n")


def _report_multi_llm_progression(model_summaries: dict):
    """Print a compact cross-model progression table."""
    if len(model_summaries) <= 1:
        return

    SIG = 0.05

    def _sig_t3(s):
        p   = s.get("t3_p_stat", 1.0)
        ab  = s.get("t3_above_naive", float("nan"))
        if isinstance(p, float) and math.isnan(p):
            return "NA"
        if p >= SIG:
            return "NS"
        if isinstance(ab, float) and math.isnan(ab):
            return "SIG"
        return "SIG" if ab > 0 else "WEAK"

    def _sig(p, baseline=0.5):
        if isinstance(p, float) and math.isnan(p):
            return "NA"
        return "SIG" if p < SIG else "NS"

    all_keys = set()
    for m_sum in model_summaries.values():
        all_keys.update(m_sum.keys())
    pm_keys  = sorted(k for k in all_keys if k[0] == "polymarket")
    man_keys = sorted(k for k in all_keys if k[0] == "manifold")
    all_keys = pm_keys + man_keys

    models = list(model_summaries.keys())
    col_w  = max(len(m) for m in models) + 2

    print(f"\n\n{'=' * 90}")
    print("  CommitBench multi-model comparison (T0->T1->T2->T3)")
    print("  SIG=strong signal  WEAK=significant but not above naive  NS=not significant  NA=no sample")
    print(f"{'=' * 90}")
    print(f"  {'topic':<28}" + "".join(
        f"  {'T0':^4}{'T1':^4}{'T2':^4}{'T3':^4}  " for _ in models))
    print(f"  {'':28}" + "".join(
        f"  {m[:col_w]:<{col_w+16}}  " for m in models))
    print("  " + "-" * 86)

    for ds, topic in all_keys:
        label = f"{ds}/{topic}"
        row = f"  {label:<28}"
        for m in models:
            s = model_summaries[m].get((ds, topic), {})
            t0 = _sig(s.get("t0_p",    float("nan")))
            t1 = _sig(s.get("t1_p_da", float("nan")))
            t2 = _sig(s.get("t2_p_4c", float("nan")))
            t3 = _sig_t3(s)
            row += f"  {t0:^4}{t1:^4}{t2:^4}{t3:^4}  "
        print(row)

    print(f"{'=' * 90}")
    print(f"  {'sig_count':<28}", end="")
    for m in models:
        m_sum = model_summaries[m]
        cnt = [0, 0, 0, 0]
        for k in all_keys:
            s = m_sum.get(k, {})
            if _sig(s.get("t0_p", float("nan"))) == "SIG":
                cnt[0] += 1
            if _sig(s.get("t1_p_da", float("nan"))) == "SIG":
                cnt[1] += 1
            if _sig(s.get("t2_p_4c", float("nan"))) == "SIG":
                cnt[2] += 1
            if _sig_t3(s) == "SIG":
                cnt[3] += 1
        print(f"  {cnt[0]:^4}{cnt[1]:^4}{cnt[2]:^4}{cnt[3]:^4}  ", end="")
    print()
    print(f"  (n={len(all_keys)} topics)")
    print(f"{'=' * 90}\n")



def _safe_mean(df, col):
    return df[col].mean() if df is not None and len(df) > 0 else float("nan")

def _safe_binom(df, col, p=0.5):
    if df is None or len(df) == 0: return 1.0
    return binom_pvalue_greater(int(df[col].sum()), len(df), p)


def model_out_tag(name: str) -> str:
    """Model name -> output directory name: qwen3:14b -> Qwen3-14B"""
    p = name.split(":")
    return p[0].capitalize() + "-" + p[1].upper() if len(p) == 2 else name.replace(":", "-")


__all__ = [_n for _n in list(globals().keys()) if not _n.startswith('__')]


