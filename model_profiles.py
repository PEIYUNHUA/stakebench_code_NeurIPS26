from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelProfile:
    alias: str
    family: str
    hf_model_id: str | None = None
    # V17: base_model_id is only used by PEFT/LoRA adapter models (family="fingpt").
    # For those models, hf_model_id is the adapter repo and base_model_id is the
    # underlying pretrained model that the adapter is applied on top of.
    base_model_id: Optional[str] = None
    use_system_prompt: bool = True
    add_no_think: bool = False
    max_new_tokens: int = 48
    temperature: float = 0.0
    do_sample: bool = False
    system_style: str = "strict_classifier"
    # prompt_format: "auto" uses chat_template or LLaMA-2 [INST] fallback.
    # "alpaca" uses "### Instruction:\n...\n\n### Response:\n" (for FinMA).
    prompt_format: str = "auto"
    # batch_size: prompts per model.generate() call per GPU replica on a GH200 (95 GB).
    # Formula: floor((95 - model_fp16_GB) / (KV_cache_per_seq_GB × 2))
    # KV per sequence ≈ n_layers × 2 × n_kv_heads × head_dim × max_ctx_tokens × 2B
    # (much smaller than model weights, so batch_size >> model_copies formula).
    # 0 = use global TRANSFORMERS_BATCH_SIZE env var default.
    batch_size: int = 0
    # multi_gpu_only: True means the model requires tensor parallelism across ALL GPUs
    # (e.g. 70B, which needs device_map="auto" spanning 2+ cards).  These models do NOT
    # use data-parallel replication — they use one copy spread across all available GPUs.
    multi_gpu_only: bool = False
    notes: List[str] = field(default_factory=list)


MODEL_PROFILES: Dict[str, ModelProfile] = {
    # ── Local / HPC (transformers / ollama) ─────────────────────────────────
    # ── batch_size: GH200 95GB, formula: (95 - model_GB) / (KV_per_seq_GB × 2) ──
    # KV per seq ≈ n_layers × 2 × n_kv_heads × head_dim × max_ctx_tokens × 2B
    # Single model copy in VRAM; KV cache is the actual limit (far smaller).
    "qwen3:8b": ModelProfile(
        alias="qwen3:8b",
        family="qwen3",
        hf_model_id="Qwen/Qwen3-8B",
        add_no_think=True,
        max_new_tokens=32,
        system_style="reasoning_classifier",
        batch_size=256,  # 16GB model; KV/seq ~210MB @ 1500 ctx; batch=256 → ~54GB KV → ~76GB total
    ),
    "qwen3:14b": ModelProfile(
        alias="qwen3:14b",
        family="qwen3",
        hf_model_id="Qwen/Qwen3-14B",
        add_no_think=True,
        max_new_tokens=32,
        system_style="reasoning_classifier",
        batch_size=192,  # 28GB model; KV/seq ~234MB @ 1500 ctx; batch=192 → ~45GB KV → ~79GB total
    ),
    "qwen3:32b": ModelProfile(
        alias="qwen3:32b",
        family="qwen3",
        hf_model_id="Qwen/Qwen3-32B",
        add_no_think=True,
        max_new_tokens=32,
        system_style="reasoning_classifier",
        batch_size=32,   # 64GB model; KV/seq ~374MB @ 1500 ctx; batch=32 → ~12GB KV → ~82GB total
    ),
    "qwen3:30b": ModelProfile(
        alias="qwen3:30b",
        family="qwen3",
        hf_model_id="Qwen/Qwen3-30B-A3B",
        add_no_think=True,
        max_new_tokens=32,
        system_style="reasoning_classifier",
        batch_size=64,   # MoE ~60GB; KV/seq ~281MB @ 1500 ctx; batch=64 → ~18GB KV → ~84GB total
    ),
    "llama3:8b": ModelProfile(
        alias="llama3:8b",
        family="llama3",
        hf_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        use_system_prompt=True,
        add_no_think=True,
        max_new_tokens=48,
        system_style="ultra_strict_label_only",
        batch_size=256,  # 16GB model; KV/seq ~189MB @ 1500 ctx; batch=256 → ~48GB KV → ~70GB total
    ),
    "llama3:70b": ModelProfile(
        alias="llama3:70b",
        family="llama3",
        hf_model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        use_system_prompt=True,
        add_no_think=True,
        max_new_tokens=64,
        system_style="ultra_strict_label_only",
        batch_size=8,        # 2×95GB=190GB; 140GB model → 44GB free; KV/seq ~900MB → batch=8 fills ~51GB KV
        multi_gpu_only=True, # needs tensor-parallel (device_map="auto") across 2 GPUs; no data replication
    ),
    "mistral:7b": ModelProfile(
        alias="mistral:7b",
        family="mistral",
        hf_model_id="mistralai/Mistral-7B-Instruct-v0.3",
        use_system_prompt=True,
        max_new_tokens=24,
        system_style="compact_classifier",
        batch_size=256,  # 14GB model; KV/seq ~186MB @ 1500 ctx; batch=256 → ~48GB KV → ~68GB total
    ),
    "gemma2:9b": ModelProfile(
        alias="gemma2:9b",
        family="gemma2",
        hf_model_id="google/gemma-2-9b-it",
        use_system_prompt=False,
        max_new_tokens=24,
        system_style="compact_classifier",
        batch_size=96,  # 18GB model; KV/seq ~535MB @ 1500 ctx (46 layers); batch=96 → ~51GB KV → ~75GB total
    ),
    "llama3.2:3b": ModelProfile(
        alias="llama3.2:3b",
        family="llama3",
        hf_model_id="meta-llama/Llama-3.2-3B-Instruct",
        use_system_prompt=True,
        add_no_think=True,
        max_new_tokens=48,
        system_style="ultra_strict_label_only",
        batch_size=384,  # 6.4GB model; KV/seq ~165MB @ 1500 ctx; batch=384 → ~63GB KV → ~76GB total
    ),
    "deepseek:r1-8b": ModelProfile(
        alias="deepseek:r1-8b",
        family="deepseek",
        hf_model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        use_system_prompt=True,
        add_no_think=False,
        max_new_tokens=512,
        system_style="reasoning_classifier",
        batch_size=192,  # 16GB model; KV/seq ~307MB (512 out + 1800 in); batch=192 → ~59GB KV → ~81GB total
    ),

    # ── Finance-domain open-source models (V18, BACKEND="transformers") ─────
    # Tier-1: instruction-tuned, proper chat template, fit on single GH200
    "qwen-finance:8b": ModelProfile(
        alias="qwen-finance:8b",
        family="qwen3",
        hf_model_id="DragonLLM/Qwen-Open-Finance-R-8B",
        use_system_prompt=True,
        add_no_think=True,
        max_new_tokens=32,
        system_style="reasoning_classifier",
        notes=[
            "V18: Qwen3-8B full fine-tune on finance corpus (DragonLLM). Apache 2.0. "
            "Uses Qwen3 chat template; add_no_think suppresses chain-of-thought tokens.",
        ],
    ),
    "fino1:8b": ModelProfile(
        alias="fino1:8b",
        family="llama3",
        hf_model_id="TheFinAI/Fino1-8B",
        use_system_prompt=True,
        add_no_think=False,
        max_new_tokens=768,
        system_style="reasoning_classifier",
        batch_size=192,  # 16GB model; KV/seq ~338MB (768 out + 1500 in); batch=192 → ~65GB KV → ~87GB total
        notes=[
            "V18: Fino1-8B — Meta-Llama-3.1-8B-Instruct SFT+RL on financial reasoning (TheFinAI). "
            "Inherits LLaMA-3.1 chat template.",
        ],
    ),
    "fino1:14b": ModelProfile(
        alias="fino1:14b",
        family="qwen25",
        hf_model_id="TheFinAI/Fino1-14B",
        use_system_prompt=True,
        add_no_think=False,
        max_new_tokens=512,
        system_style="reasoning_classifier",
        batch_size=96,   # 28GB model; KV/seq ~460MB (512 out + 1500 in); batch=96 → ~44GB KV → ~78GB total
        notes=[
            "V18: Fino1-14B — Qwen2.5-14B-Instruct fine-tuned for financial reasoning (TheFinAI). "
            "~28GB fp16; fits single GH200.",
        ],
    ),
    "finance-chat:7b": ModelProfile(
        alias="finance-chat:7b",
        family="llama2",
        hf_model_id="AdaptLLM/finance-chat",
        use_system_prompt=True,
        add_no_think=False,
        max_new_tokens=48,
        system_style="compact_classifier",
        batch_size=256,  # 14GB model; KV/seq ~189MB @ 1500 ctx; batch=256 → ~48GB KV → ~68GB total
        notes=[
            "V18: AdaptLLM Finance-Chat — LLaMA-2-Chat-7B continued pretrain + SFT on finance. "
            "ICLR 2024 (AdaptLLM). Uses LLaMA-2-Chat chat template.",
        ],
    ),

    # ── Legacy finance models (kept for reference, excluded from HPC runs) ───
    "finma-7b": ModelProfile(
        alias="finma-7b",
        family="finma",
        hf_model_id="TheFinAI/finma-7b-full",
        use_system_prompt=True,
        add_no_think=False,
        max_new_tokens=48,
        temperature=0.0,
        do_sample=False,
        system_style="ultra_strict_label_only",
        prompt_format="alpaca",
        notes=[
            "V17: FinMA-7B — LLaMA-2 7B fine-tuned on financial NLP (TheFinAI). "
            "Full-weight fine-tune via _query_transformers (no PEFT).",
        ],
    ),
    "finma-30b": ModelProfile(
        alias="finma-30b",
        family="finma",
        hf_model_id="ChanceFocus/finma-30b-nlp",
        use_system_prompt=True,
        add_no_think=False,
        max_new_tokens=48,
        temperature=0.0,
        do_sample=False,
        system_style="ultra_strict_label_only",
        prompt_format="alpaca",
        batch_size=48,   # 60GB model; KV/seq ~354MB @ 1500 ctx; batch=48 → ~17GB KV → ~83GB total
        notes=[
            "V17: FinMA-30B — LLaMA-2 30B fine-tuned on financial NLP (TheFinAI). "
            "Requires A100 80G or multi-GPU (device_map='auto').",
        ],
    ),
    "fingpt-forecaster": ModelProfile(
        alias="fingpt-forecaster",
        family="fingpt",
        hf_model_id="FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
        base_model_id="meta-llama/Llama-2-7b-chat-hf",
        use_system_prompt=True,
        add_no_think=False,
        max_new_tokens=48,
        temperature=0.0,
        do_sample=False,
        system_style="compact_classifier",
        notes=[
            "V17: FinGPT-Forecaster — LoRA adapter on LLaMA-2-7B-Chat. "
            "hf_model_id=adapter repo; base_model_id=base model. "
            "Loaded via _query_peft_lora(). Requires: pip install peft",
        ],
    ),

    # ── OpenRouter closed-source models (BACKEND="openrouter") ──────────────
    "openrouter:gpt-4o": ModelProfile(
        alias="openrouter:gpt-4o",
        family="openrouter",
        hf_model_id="openai/gpt-4o",
        use_system_prompt=True,
        max_new_tokens=64,
        temperature=0.0,
        system_style="reasoning_classifier",
    ),
    "openrouter:gpt-4o-mini": ModelProfile(
        alias="openrouter:gpt-4o-mini",
        family="openrouter",
        hf_model_id="openai/gpt-4o-mini",
        use_system_prompt=True,
        max_new_tokens=64,
        temperature=0.0,
        system_style="compact_classifier",
    ),
    "openrouter:claude-3.5-sonnet": ModelProfile(
        alias="openrouter:claude-3.5-sonnet",
        family="openrouter",
        hf_model_id="anthropic/claude-3.5-sonnet",
        use_system_prompt=True,
        max_new_tokens=64,
        temperature=0.0,
        system_style="reasoning_classifier",
    ),
    "openrouter:claude-haiku-4.5": ModelProfile(
        alias="openrouter:claude-haiku-4.5",
        family="openrouter",
        hf_model_id="anthropic/claude-haiku-4.5",
        use_system_prompt=True,
        add_no_think=False,
        max_new_tokens=64,
        temperature=0.0,
        system_style="compact_classifier",
        notes=["OpenRouter → Anthropic Claude Haiku 4.5."],
    ),
    "openrouter:gemini-2.0-flash": ModelProfile(
        alias="openrouter:gemini-2.0-flash",
        family="openrouter",
        hf_model_id="google/gemini-2.0-flash-001",
        use_system_prompt=True,
        max_new_tokens=64,
        temperature=0.0,
        system_style="compact_classifier",
    ),
    "openrouter:gemini-2.5-flash": ModelProfile(
        alias="openrouter:gemini-2.5-flash",
        family="openrouter",
        hf_model_id="google/gemini-2.5-flash",
        use_system_prompt=True,
        add_no_think=False,
        max_new_tokens=128,
        temperature=0.0,
        system_style="compact_classifier",
        notes=["OpenRouter → Google Gemini 2.5 Flash. Thinking handled server-side."],
    ),
    "openrouter:deepseek-v3": ModelProfile(
        alias="openrouter:deepseek-v3",
        family="openrouter",
        hf_model_id="deepseek/deepseek-chat-v3-0324",
        use_system_prompt=True,
        max_new_tokens=64,
        temperature=0.0,
        system_style="reasoning_classifier",
    ),
    "openrouter:deepseek-r1": ModelProfile(
        alias="openrouter:deepseek-r1",
        family="openrouter",
        hf_model_id="deepseek/deepseek-r1",
        use_system_prompt=True,
        max_new_tokens=64,
        temperature=0.0,
        system_style="reasoning_classifier",
    ),
    "openrouter:qwen3-235b": ModelProfile(
        alias="openrouter:qwen3-235b",
        family="openrouter",
        hf_model_id="qwen/qwen3-235b-a22b",
        use_system_prompt=True,
        add_no_think=True,
        max_new_tokens=64,
        temperature=0.0,
        system_style="reasoning_classifier",
    ),
    "openrouter:gpt-5.5": ModelProfile(
        alias="openrouter:gpt-5.5",
        family="openrouter",
        hf_model_id="openai/gpt-5.5",
        use_system_prompt=True,
        max_new_tokens=64,
        temperature=0.0,
        system_style="reasoning_classifier",
    ),
}


FAMILY_DEFAULTS: Dict[str, ModelProfile] = {
    "qwen3":      MODEL_PROFILES["qwen3:14b"],
    "llama3":     MODEL_PROFILES["llama3:8b"],
    "mistral":    MODEL_PROFILES["mistral:7b"],
    "gemma2":     MODEL_PROFILES["gemma2:9b"],
    "deepseek":   MODEL_PROFILES["deepseek:r1-8b"],
    "openrouter": MODEL_PROFILES["openrouter:gpt-4o"],
    # V17 Finance families
    "finma":  MODEL_PROFILES["finma-7b"],
    "fingpt": MODEL_PROFILES["fingpt-forecaster"],
}


def detect_family(model_name: str) -> str:
    name = (model_name or "").lower()
    if name.startswith("openrouter:") or name.startswith("openrouter/"):
        return "openrouter"
    if name.startswith("qwen3"):
        return "qwen3"
    if name.startswith("llama3") or "llama-3" in name:
        return "llama3"
    if name.startswith("mistral"):
        return "mistral"
    if name.startswith("gemma2") or "gemma-2" in name:
        return "gemma2"
    if name.startswith("deepseek") or "deepseek" in name:
        return "deepseek"
    # V17 Finance families
    if name.startswith("finma") or "finma" in name:
        return "finma"
    if name.startswith("fingpt") or "fingpt" in name:
        return "fingpt"
    return "generic"


def resolve_profile(model_name: str) -> ModelProfile:
    if model_name in MODEL_PROFILES:
        return MODEL_PROFILES[model_name]
    family = detect_family(model_name)
    if family in FAMILY_DEFAULTS:
        default = FAMILY_DEFAULTS[family]
        hf_model_id = os.environ.get(f"COMMITBENCH_HF_{family.upper()}")
        resolved_hf = hf_model_id or default.hf_model_id
        print(
            f"[V18][WARN] resolve_profile: '{model_name}' has no explicit MODEL_PROFILES entry. "
            f"Falling back to '{family}' family default (alias='{default.alias}', "
            f"hf='{resolved_hf}', max_new_tokens={default.max_new_tokens}, "
            f"system_style='{default.system_style}'). "
            f"Add an explicit entry in MODEL_PROFILES to suppress this warning.",
            flush=True,
        )
        return ModelProfile(
            alias=default.alias,
            family=family,
            hf_model_id=resolved_hf,
            base_model_id=default.base_model_id,  # V17: propagate for PEFT/LoRA families
            use_system_prompt=default.use_system_prompt,
            add_no_think=default.add_no_think,
            max_new_tokens=default.max_new_tokens,
            temperature=default.temperature,
            do_sample=default.do_sample,
            system_style=default.system_style,
            batch_size=default.batch_size,
            notes=default.notes,
        )
    print(
        f"[V18][WARN] resolve_profile: '{model_name}' family='{detect_family(model_name)}' "
        f"has no family default either. Using bare generic profile.",
        flush=True,
    )
    return ModelProfile(alias=model_name, family="generic", hf_model_id=model_name,
                        use_system_prompt=True)


def build_hf_model_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for alias, profile in MODEL_PROFILES.items():
        if profile.hf_model_id:
            mapping[alias] = profile.hf_model_id
    return mapping


def default_model_list() -> List[str]:
    env_model = os.environ.get("MODEL")
    env_models = os.environ.get("MODEL_LIST")
    if env_model:
        return [env_model]
    if env_models:
        return [m.strip() for m in env_models.split(",") if m.strip()]
    return ["qwen3:14b"]
