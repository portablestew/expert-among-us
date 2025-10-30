#!/bin/bash
# Expert Among Us - Run Script (Bash)
# This script runs the application using the virtual environment
# without leaving the venv activated after the script terminates

set -e

# Check if virtual environment exists
if [ ! -f ".venv/bin/python" ]; then
    echo "Error: Virtual environment not found"
    echo "Please run the appropriate install script first:"
    echo "  ./install-cpu.sh    (for CPU-only installation)"
    echo "  ./install-gpu.sh    (for GPU installation)"
    exit 1
fi

# Run the application using the venv's python directly
# This avoids activating the venv, preventing any side effects
exec .venv/bin/python -m expert_among_us "$@"