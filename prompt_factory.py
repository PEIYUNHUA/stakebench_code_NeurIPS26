from __future__ import annotations

from dataclasses import dataclass

try:
    from .model_profiles import ModelProfile
except ImportError:
    from model_profiles import ModelProfile


@dataclass(frozen=True)
class PromptBundle:
    task_id: str
    system: str
    user: str

    def to_legacy_prompt(self) -> str:
        return (
            f"[CB_TASK={self.task_id.upper()}]\n"
            f"[CB_SYSTEM]\n{self.system}\n[/CB_SYSTEM]\n"
            f"[CB_USER]\n{self.user}\n[/CB_USER]"
        )


def _system_text(profile: ModelProfile, labels: str, task_hint: str) -> str:
    if profile.system_style == "ultra_strict_label_only":
        return (
            "You are a CommitBench classifier. "
            f"Your ONLY output must be one of: {labels}. "
            "Do NOT explain. Do NOT write a sentence. Do NOT add punctuation or extra words. "
            f"Task: {task_hint}. "
            "Any response other than one of those exact labels is invalid."
        )
    if profile.system_style == "compact_classifier":
        return (
            "You are a precise label-only classifier for prediction-market research. "
            f"Valid labels: {labels}. Return one label only."
        )
    return (
        "You are a careful classifier for CommitBench, a research benchmark on "
        "prediction-market language and commitment. "
        f"Return exactly one final label from: {labels}. "
        "No chain-of-thought, no explanation, no extra text."
    )


def _family_suffix(profile: ModelProfile, labels: str) -> str:
    """Per-family format reminder appended at the end of the user message."""
    if profile.family == "llama3":
        return (
            "\n\nCRITICAL FORMAT RULE: output exactly ONE word and nothing else.\n"
            f"Permitted outputs: {labels}\n"
            "Do not write a sentence. Do not explain. Do not add punctuation.\n"
            f"Your entire response = one word from: {labels}"
        )
    if profile.family == "deepseek":
        return (
            f"\nThink privately if needed, but your visible answer must be one label only. "
            f"Final visible output: {labels}."
        )
    if profile.family == "gemma2":
        return f"\nReturn only one label: {labels}."
    # V17: Finance models (finma/fingpt) are LLaMA-2-based — same strict enforcement as llama3.
    if profile.family in ("finma", "fingpt"):
        return (
            "\n\nCRITICAL FORMAT RULE: output exactly ONE word and nothing else.\n"
            f"Permitted outputs: {labels}\n"
            "Do not write a sentence. Do not explain. Do not add punctuation.\n"
            f"Your entire response = one word from: {labels}"
        )
    return f"\nReply with exactly one label: {labels}."


# ── Task 0 ─────────────────────────────────────────────────────────────────────

def build_task0_prompt(profile: ModelProfile, question: str, text: str, zeroshot: bool = False) -> str:
    """Position detection (has_position YES/NO).

    V14 fix (§21.8.1): Llama3 now receives a concrete two-example block that
    anchors the output format, replacing the previous Llama3 zero-example path
    which produced verbose prose outputs that couldn't be parsed.
    """
    labels = "YES or NO"
    system = _system_text(profile, labels, "Task0 position detection")

    if not zeroshot:
        # V17: LLaMA-2-based families (llama3, finma, fingpt) need multi-example block.
        if profile.family in ("llama3", "finma", "fingpt"):
            example = (
                "Examples (output format: single word only):\n"
                'Comment: "I put $50 on YES because the polls moved." → YES\n'
                'Comment: "Interesting market, the data suggests yes." → NO\n'
                'Comment: "I\'m holding NO, added more yesterday." → YES\n\n'
            )
        else:
            example = (
                "Example:\n"
                'Comment: "I put $50 on YES because the polls moved."\n'
                "Label: YES\n\n"
            )
    else:
        example = ""

    user = (
        "Task: decide whether the commenter currently holds a financial position in this market.\n"
        f"Market question: {question}\n"
        f"Comment: {text}\n\n"
        "Answer YES if the comment expresses first-person ownership, stake size, buying/selling, "
        "holding, entering, trimming, or managing a position.\n"
        "Answer NO if it is only analysis, opinion, news, or discussion without first-person ownership.\n\n"
        f"{example}"
        f"{_family_suffix(profile, labels)}"
    )
    return PromptBundle(task_id="task0", system=system, user=user).to_legacy_prompt()


# ── Task 1 ─────────────────────────────────────────────────────────────────────

def build_task1_prompt(
    profile: ModelProfile, question: str, text: str, topic: str = ""
) -> str:
    """Belief inference (YES/NO/NEUTRAL).

    V14 additions:
    - topic parameter: injects domain-specific guidance when topic contains
      'crypto' to address §21.8.2 (Qwen3-32B poly/crypto T1 3c=0.442, below
      random baseline 0.500, p=0.999).
    - Llama3 fix (§21.8.1): explicit anti-NEUTRAL instruction + few-shot
      examples showing weak signals should resolve to YES/NO, not NEUTRAL.

    V15 additions:
    - Stronger crypto domain note: V14 note was insufficient for Qwen3-32B
      poly/crypto (§22.8.2, 3c=0.338 p=0.999 in res425). Added explicit
      anti-reversal examples and FUD/sarcasm disambiguation guidance.
    """
    labels = "YES, NO, or NEUTRAL"
    system = _system_text(profile, labels, "Task1 belief inference")

    # Crypto sarcasm/contrarian note (§21.8.2, strengthened in V15 for §22.8.2)
    domain_note = ""
    if "crypto" in topic.lower():
        domain_note = (
            "\nDomain note (Crypto): Cryptocurrency prediction-market commenters frequently "
            "use irony, sarcasm, FUD, and contrarian framing that can REVERSE the apparent "
            "surface sentiment. Apply the following rules strictly:\n"
            "  1. 'Obviously going to zero' / 'this is done' written by a YES-position holder "
            "     = the commenter BELIEVES YES (sarcastic bearish framing).\n"
            "  2. Explicit bearish language ('dump', 'crash', 'rug') in a comment whose author "
            "     holds a NO position = commenter BELIEVES NO (consistent).\n"
            "  3. 'FUD' accusations by a commenter = commenter believes the negative narrative "
            "     is false, so they lean YES.\n"
            "  4. Always infer the UNDERLYING MARKET BELIEF from position + semantic context, "
            "     not surface tone alone. Do NOT reverse your label just because the tone is "
            "     negative — check whether the negativity is sarcastic or sincere.\n"
        )

    # V17: LLaMA-2-based families (llama3, finma, fingpt) share the anti-NEUTRAL fix (§21.8.1)
    if profile.family in ("llama3", "finma", "fingpt"):
        extra = (
            "\nIMPORTANT — NEUTRAL is EXTREMELY RARE:\n"
            "NEUTRAL is ONLY for comments about market mechanics with zero directional "
            "content (e.g. 'when does this resolve?', 'what counts as a win?').\n"
            "ANY hint of belief, expectation, doubt, or opinion = YES or NO.\n\n"
            "Examples (single-word answer ONLY):\n"
            'Q: "Will the economy grow?" Comment: "Not sure it can happen this year." → NO\n'
            'Q: "Will X win?" Comment: "The polls look promising for X." → YES\n'
            'Q: "Will rates rise?" Comment: "I think rates will probably stay flat." → NO\n'
            'Q: "Will Y happen?" Comment: "What are the resolution criteria?" → NEUTRAL\n\n'
        )
    else:
        extra = (
            "\nRules:\n"
            "  - Weak or hedged language still counts as YES or NO.\n"
            "  - Skepticism or doubt about the YES outcome → NO.\n"
            "  - NEUTRAL is reserved ONLY for zero-stance procedural comments.\n\n"
        )

    user = (
        "Task: infer the direction the commenter believes this market will resolve.\n"
        f"Market question: {question}\n"
        f"Comment: {text}\n\n"
        "YES = commenter implies the market resolves YES.\n"
        "NO = commenter implies the market resolves NO.\n"
        "NEUTRAL = ONLY when comment is purely procedural with absolutely zero directional stance.\n"
        f"{domain_note}"
        f"{extra}"
        f"{_family_suffix(profile, labels)}"
    )
    return PromptBundle(task_id="task1", system=system, user=user).to_legacy_prompt()


# ── Task 2 ─────────────────────────────────────────────────────────────────────

def build_task2_prompt(
    profile: ModelProfile, question: str, text: str, position_side: str, position_size: float
) -> str:
    """4-class action prediction (flip/increase/decrease/hold).

    V14 improvement (§21.8.3): enriched with concrete examples and detailed
    signal descriptions to address the T2-4c size-reversal where Qwen3-32B
    macro acc=0.161 (0/9 significant) fell below Qwen3-14B.
    """
    labels = "flip, increase, decrease, or hold"
    system = _system_text(profile, labels.upper(), "Task2 language-action consistency")
    opposite = "NO" if str(position_side).upper() == "YES" else "YES"

    user = (
        "Task: predict the most likely position action in the next 48 hours.\n"
        f"Market question: {question}\n"
        f"Current position: {position_side} (${position_size:,.0f})\n"
        f"Comment: {text}\n\n"
        "Signal descriptions:\n"
        f"  flip     = comment NOW favors {opposite}, the OPPOSITE of the current {position_side} position.\n"
        f"             Signals: changed mind, new evidence contradicts position, 'I was wrong', "
        f"'switching to {opposite}'\n"
        f"  increase = STRONGER conviction in {position_side}.\n"
        f"             Signals: adding more, doubling down, very confident, 'loading up', "
        f"escalating bullish/bearish\n"
        "  decrease = DOUBT, WORRY, or HEDGING about the current position.\n"
        "             Signals: not as confident, risk too high, taking profit, 'trimming', cautious\n"
        "  hold     = neutral observation, no change in conviction.\n"
        "             Signals: just watching, reiterating view without escalation or new doubt\n\n"
        "IMPORTANT: Do NOT default to hold.\n"
        "  Any doubt signal → decrease\n"
        "  Any opposite-side signal → flip\n\n"
        "Examples:\n"
        f"  Position: {position_side} ($500).  "
        f"Comment: 'The latest data makes me nervous, might be overexposed.'  → decrease\n"
        f"  Position: {position_side} ($200).  "
        f"Comment: 'Everything confirms my view, very confident now.'  → increase\n"
        f"  Position: {position_side} ($100).  "
        f"Comment: 'Actually I think this resolves {opposite} now.'  → flip\n"
        f"  Position: {position_side} ($300).  "
        f"Comment: 'Interesting development, just watching for now.'  → hold\n\n"
        f"{_family_suffix(profile, labels)}"
    )
    return PromptBundle(task_id="task2", system=system, user=user).to_legacy_prompt()


def build_task2_flip_hold_prompt(
    profile: ModelProfile, question: str, text: str, position_side: str, position_size: float
) -> str:
    labels = "flip or hold"
    system = _system_text(profile, labels.upper(), "Task2 binary flip-vs-hold")
    opposite = "NO" if str(position_side).upper() == "YES" else "YES"
    user = (
        "Task: binary classification for action prediction.\n"
        f"Market question: {question}\n"
        f"Current position: {position_side} (${position_size:,.0f})\n"
        f"Comment: {text}\n\n"
        f"flip = the comment clearly favors {opposite} (OPPOSITE of the current {position_side} position).\n"
        "hold = everything else — reaffirming, mild doubt, neutral, or only partially doubtful.\n\n"
        f"{_family_suffix(profile, labels)}"
    )
    return PromptBundle(task_id="task2_flip_hold", system=system, user=user).to_legacy_prompt()


def build_task2_decrease_hold_prompt(
    profile: ModelProfile, question: str, text: str, position_side: str, position_size: float
) -> str:
    labels = "decrease or hold"
    system = _system_text(profile, labels.upper(), "Task2 binary decrease-vs-hold")
    user = (
        "Task: binary classification for action prediction.\n"
        f"Market question: {question}\n"
        f"Current position: {position_side} (${position_size:,.0f})\n"
        f"Comment: {text}\n\n"
        f"decrease = DOUBT, CAUTION, WORRY, or PROFIT-TAKING about the {position_side} position.\n"
        "           Signals: 'not as confident', 'taking profit', 'trimming', 'too risky', hedging.\n"
        "hold     = neutral, observational, or re-affirms conviction without doubt.\n\n"
        f"{_family_suffix(profile, labels)}"
    )
    return PromptBundle(task_id="task2_decrease_hold", system=system, user=user).to_legacy_prompt()


# ── Task 3 ─────────────────────────────────────────────────────────────────────

def build_task3_prompt(profile: ModelProfile, question: str, comments_text: str, current_odds: float) -> str:
    labels = "UP, DOWN, or NEUTRAL"
    system = _system_text(profile, labels, "Task3 belief-price lead")
    user = (
        "Task: predict the next-window direction of the market YES probability.\n"
        f"Market question: {question}\n"
        f"Current YES probability: {current_odds:.2f}\n"
        f"Recent comments:\n{comments_text}\n\n"
        "UP = YES probability likely increases next.\n"
        "DOWN = YES probability likely decreases next.\n"
        "NEUTRAL = direction is genuinely unclear from the comments.\n\n"
        f"{_family_suffix(profile, labels)}"
    )
    return PromptBundle(task_id="task3", system=system, user=user).to_legacy_prompt()
