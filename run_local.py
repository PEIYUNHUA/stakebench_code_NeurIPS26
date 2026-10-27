"""
run_local.py — CommitBench V18 entry point

V18 = V16 local improvements (parallel inference, bool fixes, state isolation)
    + V17 finance models (finma/fingpt + PEFT)

BACKEND options:
    "transformers" — local/HPC GPU for open-source models (qwen3, llama3, finma, fingpt, etc.)
    "openrouter"   — OpenRouter API for closed-source models (claude-haiku-4.5, gemini-2.5-flash, etc.)

TASK options:
    "task01"  — run Task0 (position detection) + Task1 (belief inference) only
    "task2"   — run Task2 (language-action consistency) only
    "task3"   — run Task3 (belief-price lead) only
    "all"     — run all tasks in sequence
"""
from __future__ import annotations
import os
import sys

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit only this section; no changes needed in other files
# ════════════════════════════════════════════════════════════════════════════

# Task to run
TASK = os.environ.get("TASK", "all")

# Backend selection
#   "transformers" — local/HPC GPU (Finance models and other open-source models)
#   "openrouter"   — OpenRouter API (closed-source models)
BACKEND = os.environ.get("BACKEND", "transformers")

# Model alias (key in MODEL_PROFILES)
# Open-source options: qwen3:14b / qwen3:32b / llama3:8b / mistral:7b / gemma2:9b
#                      deepseek:r1-8b / finma-7b / finma-30b / fingpt-forecaster
# Closed-source options (requires BACKEND="openrouter"):
#              openrouter:claude-haiku-4.5 / openrouter:gemini-2.5-flash
#              openrouter:gpt-4o / openrouter:deepseek-v3
MODEL = os.environ.get("MODEL", "deepseek:r1-8b")

# OpenRouter API Key (required when BACKEND="openrouter")
# Leave empty to read from environment variable OPENROUTER_API_KEY
OPENROUTER_API_KEY = ""

# Parallel threads (only used when BACKEND="openrouter")
OPENROUTER_CONCURRENCY = 8

# Local data paths (override via environment variables for HPC, see run_hpc.sh)
MANIFOLD_DIR   = os.environ.get(
    "COMMITBENCH_MANIFOLD_DIR",
    "/path/to/workdir/data/manifold_data"
)
POLYMARKET_DIR = os.environ.get(
    "COMMITBENCH_POLYMARKET_DIR",
    "/path/to/workdir/data/polymarket_data"
)
OUT_DIR = os.environ.get(
    "COMMITBENCH_OUT_DIR",
    "/path/to/workdir/result/commitbench_v18"
)

# ════════════════════════════════════════════════════════════════════════════
#  No changes needed below this line
# ════════════════════════════════════════════════════════════════════════════

os.environ["BACKEND"]                    = BACKEND
os.environ["MODEL"]                      = MODEL
os.environ["COMMITBENCH_MANIFOLD_DIR"]   = MANIFOLD_DIR
os.environ["COMMITBENCH_POLYMARKET_DIR"] = POLYMARKET_DIR
os.environ["COMMITBENCH_OUT_DIR"]        = OUT_DIR
os.environ["OPENROUTER_CONCURRENCY"]     = str(OPENROUTER_CONCURRENCY)
os.environ["PYTHONIOENCODING"]           = "utf-8"
if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Main pipeline ────────────────────────────────────────────────────────────
from pipeline_runners import run_task01_pipeline, run_task2_pipeline, run_task3_pipeline

if TASK in ("task01", "all"):
    run_task01_pipeline()

if TASK in ("task2", "all"):
    run_task2_pipeline()

if TASK in ("task3", "all"):
    run_task3_pipeline()
