from __future__ import annotations
# CommitBench V18
# Merged from V16 (local improvements) + V17 (Finance models).
# Key changes over V17:
#   - Carry forward all V16 local fixes:
#       * parallel inference (_run_infer_loop / OPENROUTER_CONCURRENCY)
#       * bool-column normalisation in _csv_load_partial and _report_*
#       * model-prefixed run_state.json keys (MODEL/ds/topic/task)
#       * thread-safe audit log (_audit_lock)
#       * robust OpenRouter error handling (non-JSON, empty choices, last_status)
#       * OpenRouter model profiles: claude-haiku-4.5, gemini-2.5-flash
#   - Carry forward all V17 additions:
#       * Finance open-source models: finma-7b, finma-30b, fingpt-forecaster
#       * PEFT/LoRA backend (_query_peft_lora) for fingpt family
#       * base_model_id field in ModelProfile

import os


BASE_DIR = os.environ.get("COMMITBENCH_BASE_DIR",
                          "/path/to/workdir/v18")
PROJECT_ROOT = os.environ.get("COMMITBENCH_PROJECT_ROOT",
                               "/path/to/workdir")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek:r1-8b"
TEMPERATURE = 0.0
TIMEOUT = 180
BACKEND = os.environ.get("BACKEND", "transformers")

HF_CACHE_DIR = os.environ.get("HF_HOME",
                               "/path/to/workdir/llm_cache/hf")
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

HF_MODEL_MAP = {
    # Qwen3
    "qwen3:30b": "Qwen/Qwen3-30B-A3B",
    "qwen3:32b": "Qwen/Qwen3-32B",
    "qwen3:14b": "Qwen/Qwen3-14B",
    "qwen3:8b":  "Qwen/Qwen3-8B",
    # Llama3
    "llama3:70b":   "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama3:8b":    "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.2:3b":  "meta-llama/Llama-3.2-3B-Instruct",
    # DeepSeek-R1 distilled
    "deepseek:r1-8b":  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # V17 Finance models
    "finma-7b":          "TheFinAI/finma-7b-full",
    "finma-30b":         "ChanceFocus/finma-30b-nlp",
    "fingpt-forecaster": "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
}

# ── Data paths ──────────────────────────────────────────────────────────────
_HPC_MANIFOLD   = "/path/to/workdir/data/manifold_data"
_HPC_POLYMARKET = "/path/to/workdir/data/polymarket_data"
_HPC_OUT        = "/path/to/workdir/result/commitbench_v18"

DATA_ROOTS = {
    "manifold":   os.environ.get("COMMITBENCH_MANIFOLD_DIR",   _HPC_MANIFOLD),
    "polymarket": os.environ.get("COMMITBENCH_POLYMARKET_DIR", _HPC_POLYMARKET),
}
PRICES_DIRS = {ds: os.path.join(root, "prices") for ds, root in DATA_ROOTS.items()}
OUT_DIR = os.environ.get("COMMITBENCH_OUT_DIR", _HPC_OUT)
RUN_FULL = False

N_WINDOWS_TARGET = 8
MIN_COMMENTS_PER_WINDOW = 10
MIN_WINDOWS_PASS = 2
TASK3_MIN_WINDOW_COMMENTS = 2
TASK3_PRICE_EPS = 0.005
TASK3_NEAR50_LOWER = 0.30
TASK3_NEAR50_UPPER = 0.70

N_MARKETS_PER_TOPIC = {
    "polymarket": 20,
    "manifold": 10,
}

TOPIC_CONFIG = {
    "tech_other":    {"n_task0": 80,  "n_task1": 150, "n_task2": 60, "n_task3": 40},
    "politics_us":   {"n_task0": 80,  "n_task1": 150, "n_task2": 60, "n_task3": 40},
    "politics_intl": {"n_task0": 50,  "n_task1": 100, "n_task2": 40, "n_task3": 20},
    "crypto":        {"n_task0": 50,  "n_task1": 100, "n_task2": 40, "n_task3": 20},
    "meta":          {"n_task0": 30,  "n_task1": 60,  "n_task2": 25, "n_task3": 12},
    "social":        {"n_task0": 30,  "n_task1": 60,  "n_task2": 25, "n_task3": 12},
    "other":         {"n_task0": 30,  "n_task1": 60,  "n_task2": 25, "n_task3": 12},
    "culture":       {"n_task0": 30,  "n_task1": 60,  "n_task2": 25, "n_task3": 12},
    "tech_ai":       {"n_task0": 30,  "n_task1": 60,  "n_task2": 25, "n_task3": 12},
    "science":       {"n_task0": 25,  "n_task1": 50,  "n_task2": 20, "n_task3": 10},
    "sports_other":  {"n_task0": 20,  "n_task1": 40,  "n_task2": 15, "n_task3": 8},
    "economics":     {"n_task0": 30,  "n_task1": 60,  "n_task2": 25, "n_task3": 12},
    "sports_nba":    {"n_task0": 30,  "n_task1": 60,  "n_task2": 25, "n_task3": 12},
    "sports_mlb":    {"n_task0": 30,  "n_task1": 60,  "n_task2": 25, "n_task3": 12},
    "weather":       {"n_task0": 25,  "n_task1": 50,  "n_task2": 20, "n_task3": 10},
}
DEFAULT_TOPIC_CFG = {"n_task0": 30, "n_task1": 60, "n_task2": 25, "n_task3": 12}

TOPIC_CONFIG_POLYMARKET = {
    "sports_nba":    {"n_task0": 150, "n_task1": 400, "n_task2": 150, "n_task3": 100},
    "sports_mlb":    {"n_task0": 150, "n_task1": 400, "n_task2": 150, "n_task3": 100},
    "crypto":        {"n_task0": 150, "n_task1": 300, "n_task2": 120, "n_task3": 80},
    "politics_us":   {"n_task0": 80,  "n_task1": 150, "n_task2": 60,  "n_task3": 40},
    "tech_other":    {"n_task0": 60,  "n_task1": 120, "n_task2": 50,  "n_task3": 25},
    "politics_intl": {"n_task0": 60,  "n_task1": 120, "n_task2": 50,  "n_task3": 25},
    "weather":       {"n_task0": 50,  "n_task1": 100, "n_task2": 40,  "n_task3": 20},
    "economics":     {"n_task0": 40,  "n_task1": 80,  "n_task2": 30,  "n_task3": 15},
    "sports_other":  {"n_task0": 40,  "n_task1": 80,  "n_task2": 30,  "n_task3": 15},
    "other":         {"n_task0": 20,  "n_task1": 40,  "n_task2": 15,  "n_task3": 8},
    "tech_ai":       {"n_task0": 30,  "n_task1": 60,  "n_task2": 25,  "n_task3": 12},
    "science":       {"n_task0": 25,  "n_task1": 50,  "n_task2": 20,  "n_task3": 10},
    "culture":       {"n_task0": 25,  "n_task1": 50,  "n_task2": 20,  "n_task3": 10},
    "social":        {"n_task0": 25,  "n_task1": 50,  "n_task2": 20,  "n_task3": 10},
    "meta":          {"n_task0": 25,  "n_task1": 50,  "n_task2": 20,  "n_task3": 10},
}
DEFAULT_TOPIC_CFG_POLYMARKET = {"n_task0": 40, "n_task1": 80, "n_task2": 30, "n_task3": 15}

MIN_TEXT_LEN = 30
HOLD_HOURS = 48
ACTION_THR = 0.10
PRICE_EPS = 0.01
SEED = 42
CONSENSUS_STRENGTH_THR = 0.15
MIN_EXPLICIT_TASK0 = 5

MODEL_ALL = [
    # General-purpose instruction models
    "deepseek:r1-8b",
    "qwen3:8b",
    "qwen3:14b",
    "qwen3:30b",
    "qwen3:32b",
    "llama3.2:3b",
    "llama3:8b",
    "llama3:70b",
    "mistral:7b",
    "gemma2:9b",
    # Finance-domain fine-tuned models (V18)
    "finma-30b",
    "fino1:8b",
    "fino1:14b",
    "finance-chat:7b",
]
MODEL_LIST = [os.environ["MODEL"]] if os.environ.get("MODEL") else MODEL_ALL
TOPIC_NO_CAP = {"economics"}
EXPLICIT_QUALITY_THR = 1
SPORTS_TOPICS_ZERO_THR = {"sports_nba", "sports_mlb", "sports_other"}
N_COMMENT_GROUPS = 8
TASK2_HOLD_HOURS_LIST = [6, 12, 24, 48, 72]

# ── OpenRouter API (BACKEND="openrouter") ────────────────────────────────────
# Always set via environment variable — no hardcoded key for HPC safety.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_SITE_URL = "https://github.com/PEIYUNHUA/stakebench_code_NeurIPS26"
OPENROUTER_SITE_NAME = "CommitBench"
OPENROUTER_TIMEOUT = 120
OPENROUTER_MAX_RETRIES = 5
OPENROUTER_RETRY_BASE_DELAY = 2.0
# Parallel threads for OpenRouter calls (ignored for transformers/ollama backends).
OPENROUTER_CONCURRENCY = int(os.environ.get("OPENROUTER_CONCURRENCY", "8"))

# ── Transformers batch inference (BACKEND="transformers") ────────────────────
# Number of prompts per model.generate() call. With 100 GB VRAM and an 8B
# fp16 model (~16 GB), the remaining ~84 GB can hold large KV caches.
# Increase if GPU utilisation is low; decrease if you get OOM on long prompts.
TRANSFORMERS_BATCH_SIZE = int(os.environ.get("TRANSFORMERS_BATCH_SIZE", "32"))

# ── vLLM multi-GPU backend (BACKEND="vllm") ──────────────────────────────────
# tensor_parallel_size: 0 = auto-detect (use all visible GPUs).
VLLM_TENSOR_PARALLEL_SIZE = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "0"))
# Fraction of GPU memory vLLM pre-allocates for the KV cache.
VLLM_GPU_MEMORY_UTIL = float(os.environ.get("VLLM_GPU_MEMORY_UTIL", "0.90"))
# Maximum prompts submitted to vLLM in one generate() call.
VLLM_BATCH_SIZE = int(os.environ.get("VLLM_BATCH_SIZE", "512"))
