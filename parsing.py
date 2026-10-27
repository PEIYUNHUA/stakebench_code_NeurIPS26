from __future__ import annotations

import re
from typing import Iterable


TASK_LABELS = {
    "task0": ("YES", "NO"),
    "task1": ("YES", "NO", "NEUTRAL"),
    "task2": ("FLIP", "INCREASE", "DECREASE", "HOLD"),
    "task2_flip_hold": ("FLIP", "HOLD"),
    "task2_decrease_hold": ("DECREASE", "HOLD"),
    "task3": ("UP", "DOWN", "NEUTRAL"),
}

# V16 fix: strip the ENTIRE <think>…</think> block (content + tags).
# V15 only stripped the tags with r"</?think>", leaving the full reasoning
# chain in the text — causing label extraction to fire on thinking content
# instead of the final answer (root cause of DeepSeek-R1 T0/T2-4c/T3 ≈ 0).
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def infer_task_id(prompt: str) -> str:
    match = re.search(r"CB_TASK\s*=\s*([A-Z0-9_]+)", prompt)
    if match:
        return match.group(1).lower()
    lower = prompt.lower()
    if "flip / increase / decrease / hold" in lower:
        return "task2"
    if "flip / hold" in lower:
        return "task2_flip_hold"
    if "decrease / hold" in lower:
        return "task2_decrease_hold"
    if "up (probability will increase)" in lower:
        return "task3"
    if "yes / no / neutral" in lower:
        return "task1"
    return "task0"


def normalize_label(raw: str, task_id: str) -> str:
    """Extract a valid label from LLM output.

    Handles both terse single-word outputs (Qwen3-style) and verbose reasoning
    outputs (Llama3-style) that bury the answer in surrounding text.

    Resolution order:
      1. Exact fullmatch (terse models, single-word answer)
      2. First token stripped of punctuation
      3. Explicit answer-marker context ("Answer: YES", "The label is NO")
      4. Last line of response (conclusion-last models)
      5. Last standalone word-boundary occurrence, with negation guard
      6. Synonym expansion for belief tasks
    """
    labels = TASK_LABELS.get(task_id, ())
    if not raw:
        return "UNKNOWN"

    # Normalize GPT-2/DeepSeek-style byte-level BPE escape chars to ASCII equivalents.
    # These chars are Unicode letters so they break \b word-boundary matching.
    # Ġ (U+0120) = space (0x20), Ċ (U+010A) = newline (0x0A).
    raw = raw.replace('Ġ', ' ').replace('Ċ', '\n')

    # V16: strip full <think>…</think> block BEFORE any processing.
    # DeepSeek-R1-Distill fix: <think> is a special token stripped by the tokenizer,
    # but </think> remains as plain text. Take everything after the last </think>.
    if "</think>" in raw and "<think>" not in raw:
        raw = raw.split("</think>")[-1]
    text = _THINK_BLOCK_RE.sub("", raw.strip()).strip()
    text = re.sub(r"\s+", " ", text).strip()
    upper = text.upper()

    # 1. Exact match
    for label in labels:
        if re.fullmatch(rf"[^\w]*{re.escape(label)}[^\w]*", upper):
            return label

    # 2. First token stripped of surrounding punctuation
    first_token = upper.split()[0].strip(".,!?:;-") if upper.split() else ""
    if first_token in labels:
        return first_token

    # 3. Explicit answer-marker ("Answer: YES", "My answer is NO", "## Final Response\n\nYES")
    # FINAL RESPONSE covers fino1-style "## Thinking ... ## Final Response\n\n[LABEL]" output.
    answer_m = re.search(
        r"(?:ANSWER|LABEL|FINAL ANSWER|FINAL RESPONSE|MY ANSWER|THE ANSWER IS|OUTPUT|CLASSIFY AS|I(?:'D| WOULD) SAY)"
        r"\s*[:\-=]?\s*([A-Z]+)",
        upper,
    )
    if answer_m:
        candidate = answer_m.group(1).strip(".,!?:;")
        if candidate in labels:
            return candidate

    # 4. Last line of response (conclusion-last verbose models)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        last_upper = lines[-1].upper()
        for label in labels:
            if re.fullmatch(rf"[^\w]*{re.escape(label)}[^\w]*", last_upper):
                return label
        last_first = last_upper.split()[0].strip(".,!?:;-") if last_upper.split() else ""
        if last_first in labels:
            return last_first

    # 5. Last standalone word-boundary occurrence with simple negation guard.
    #    Using last (not first) handles "I can't say YES or NO so I'll go with NEUTRAL".
    last_found: str = ""
    last_pos = -1
    for label in labels:
        for m in re.finditer(rf"\b{re.escape(label)}\b", upper):
            if m.start() > last_pos:
                before = upper[max(0, m.start() - 6) : m.start()]
                if re.search(r"\bNOT\b", before):
                    continue
                last_pos = m.start()
                last_found = label
    if last_found:
        return last_found

    # 6. Synonym expansion for binary belief tasks
    if task_id in {"task0", "task1"}:
        if any(tok in upper for tok in ["BULLISH", "LEAN YES", "RESOLVE YES", "HOLDS YES"]):
            return "YES"
        if any(tok in upper for tok in ["BEARISH", "LEAN NO", "RESOLVE NO", "HOLDS NO"]):
            return "NO"

    return "UNKNOWN"


def allowed_labels(task_id: str) -> Iterable[str]:
    return TASK_LABELS.get(task_id, ())
