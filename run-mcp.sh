#!/bin/bash
set -e
cd "$(dirname "$0")"

# Auto-install if venv doesn't exist
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Running install-gpu.sh..."
    ./install-gpu.sh
fi

exec .venv/bin/python -m expert_among_us.__mcp__ "$@"