#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot environment setup for RunPod (or any fresh Linux GPU box)
# =============================================================================
# Usage:
#   bash setup.sh
#
# After setup:
#   source .venv/bin/activate
#   wandb login
#   uv run python train.py --exp experiments/exp01_post_norm.yaml
# =============================================================================

set -euo pipefail

echo "============================================"
echo " Story-LLM — RunPod Environment Setup"
echo "============================================"

# --- 1. Install uv if not present -------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "[1/4] Installing uv..."
    pip install uv -q
else
    echo "[1/4] uv already installed ($(uv --version))"
fi

# --- 2. Create virtual environment ------------------------------------------
echo "[2/4] Creating .venv..."
uv venv --python 3.11
source .venv/bin/activate

# --- 3. Install PyTorch with CUDA -------------------------------------------
# Detect CUDA version and pick the matching wheel index.
# Adjust cu124 → cu118 / cu121 if your RunPod image uses a different CUDA.
echo "[3/4] Installing PyTorch (CUDA 12.4)..."
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
uv pip install torch --index-url "$TORCH_INDEX" -q

# --- 4. Install project + remaining dependencies ----------------------------
echo "[4/4] Installing story-llm and dependencies..."
uv pip install -e "." -q

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  wandb login                   # paste your API key"
echo "  uv run python train.py --exp experiments/exp01_post_norm.yaml"
echo ""
