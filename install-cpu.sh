#!/bin/bash
set -e

echo "================================================"
echo "Expert Among Us - CPU Installation"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Error: Python 3 is not installed"
    echo "Please install Python 3.12 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.12"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "[ERROR] Error: Python $PYTHON_VERSION found, but Python 3.12+ is required"
    exit 1
fi

echo "[OK] Python $PYTHON_VERSION detected"
echo ""

# Check for uv
echo "Checking for uv package manager..."
if ! command -v uv &> /dev/null; then
    echo "[INSTALL] Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    if ! command -v uv &> /dev/null; then
        echo "[ERROR] Error: uv installation failed"
        echo "Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
else
    echo "[OK] uv is already installed"
fi

echo ""
echo "Installing project dependencies..."
uv sync

echo ""
echo "================================================"
echo "[OK] Installation Complete!"
echo "================================================"
echo ""
echo "To run the application, use:"
echo "  uv run python -m expert_among_us --help"
echo ""
echo "Or activate the virtual environment:"
echo "  source .venv/bin/activate"
echo "  python -m expert_among_us --help"
echo ""
echo "[INFO] For more information, see README.md"
echo ""