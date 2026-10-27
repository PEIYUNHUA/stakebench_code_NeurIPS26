#!/bin/bash
# StakeBench -- SLURM job script for the open-weight models (transformers backend)
#
# Usage:
#   Single model:  sbatch --export=MODEL=qwen3:14b run_hpc.sh
#   All 12 models: sbatch --array=0-11 run_hpc.sh
#
# Model array index mapping:
#   0: deepseek:r1-8b
#   1: qwen3:8b
#   2: qwen3:14b
#   3: qwen3:30b
#   4: qwen3:32b
#   5: llama3.2:3b
#   6: mistral:7b
#   7: gemma2:9b
#   8: finma-30b
#   9: fino1:8b          (LLaMA-3.1-8B, finance fine-tune)
#  10: fino1:14b         (Qwen2.5-14B, finance fine-tune)
#  11: finance-chat:7b   (LLaMA-2-Chat-7B, finance fine-tune)

#SBATCH --job-name=commitbench_v18
#SBATCH --output=/path/to/workdir/logs/v18_%A_%a.out
#SBATCH --error=/path/to/workdir/logs/v18_%A_%a.err
#SBATCH --gpus=4
#SBATCH --time=24:00:00

# ── Model selection ───────────────────────────────────────────────────────────
MODEL_ARRAY=(
    "deepseek:r1-8b"    # 0
    "qwen3:8b"          # 1
    "qwen3:14b"         # 2
    "qwen3:30b"         # 3
    "qwen3:32b"         # 4
    "llama3.2:3b"       # 5
    "mistral:7b"        # 6
    "gemma2:9b"         # 7
    "finma-30b"         # 8
    "fino1:8b"          # 9
    "fino1:14b"         # 10
    "finance-chat:7b"   # 11
)

# Use SLURM_ARRAY_TASK_ID if running as array job; fallback to single MODEL env var.
if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    export MODEL="${MODEL_ARRAY[$SLURM_ARRAY_TASK_ID]}"
fi
# If MODEL is still unset (non-array single job), default to deepseek:r1-8b
export MODEL="${MODEL:-deepseek:r1-8b}"

# ── Environment ───────────────────────────────────────────────────────────────
module load cray-python/3.11.7
source /path/to/workdir/venv/bin/activate

export BACKEND="transformers"
export TASK="all"

export COMMITBENCH_MANIFOLD_DIR="/path/to/workdir/data/manifold_data"
export COMMITBENCH_POLYMARKET_DIR="/path/to/workdir/data/polymarket_data"
export COMMITBENCH_OUT_DIR="/path/to/workdir/result/commitbench_v18"
export HF_HOME="/path/to/workdir/llm_cache/hf"
export TRANSFORMERS_CACHE="${HF_HOME}"

# OpenRouter key (only needed for BACKEND=openrouter jobs)
# export OPENROUTER_API_KEY="sk-or-v1-..."

export PYTHONIOENCODING="utf-8"
export TOKENIZERS_PARALLELISM="false"
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")}"
# Number of model replicas to load per GPU (default 1).
# e.g. REPLICAS_PER_GPU=5 loads 5 independent copies of an 8B model on each GH200
# (5 x 16 GB = 80 GB), running 5 threads per GPU simultaneously.
# Total worker threads = gpus x REPLICAS_PER_GPU (e.g. 4 GPUs x 5 = 20 threads).
export REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-1}"

# Per-replica batch size.
# REPLICAS_PER_GPU=1 (default): profile.batch_size takes priority (large, fills VRAM via KV cache).
# REPLICAS_PER_GPU>1: TRANSFORMERS_BATCH_SIZE is used directly; profile.batch_size is bypassed
#   because it was calibrated for a single model copy.  Auto-default to 8 to avoid OOM.
if [ "${REPLICAS_PER_GPU}" -gt 1 ]; then
    export TRANSFORMERS_BATCH_SIZE="${TRANSFORMERS_BATCH_SIZE:-8}"
else
    export TRANSFORMERS_BATCH_SIZE="${TRANSFORMERS_BATCH_SIZE:-32}"
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="/path/to/workdir/v18"
mkdir -p "${COMMITBENCH_OUT_DIR}" /path/to/workdir/logs

echo "=========================================="
echo "CommitBench V18 -- SLURM job"
echo "MODEL:            ${MODEL}"
echo "BACKEND:          ${BACKEND}"
echo "TASK:             ${TASK}"
echo "OUT_DIR:          ${COMMITBENCH_OUT_DIR}"
echo "REPLICAS_PER_GPU: ${REPLICAS_PER_GPU}"
echo "BATCH_SIZE:       ${TRANSFORMERS_BATCH_SIZE}"
echo "NODE:             $(hostname)"
echo "GPUS:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

cd "${SCRIPT_DIR}"
python run_local.py
