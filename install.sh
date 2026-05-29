#!/bin/bash
set -e

# Expert Among Us - Local Development Setup
# For end-users: install with `uv tool install expert-among-us` — see README.

echo "Expert Among Us - Local Development Setup"
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Syncing dependencies (CUDA-enabled PyTorch)..."
uv sync

echo ""
echo "Verifying install..."
uv run expert-among-us --help

echo ""
echo "Done. Use 'uv run expert-among-us' to run from this clone."
