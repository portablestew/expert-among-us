#!/bin/bash
set -e

echo "================================================"
echo "Expert Among Us - GPU Installation"
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
echo "Installing base project dependencies..."
uv sync

echo ""
echo "Activating virtual environment and installing GPU-enabled PyTorch..."
echo ""

# Activate virtual environment and install GPU PyTorch
source .venv/bin/activate

echo "Installing PyTorch with CUDA support..."
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

echo ""
echo "Verifying GPU detection..."
python3 -c "import torch; cuda_available = torch.cuda.is_available(); print(f'CUDA available: {cuda_available}'); exit(0 if cuda_available else 1)" || {
    echo ""
    echo "[WARNING] WARNING: CUDA is not available"
    echo "This could mean:"
    echo "  1. You don't have an NVIDIA GPU"
    echo "  2. NVIDIA drivers are not installed"
    echo "  3. CUDA toolkit is not properly configured"
    echo ""
    echo "To check your GPU and drivers:"
    echo "  nvidia-smi"
    echo ""
    echo "The installation will continue, but you'll be using CPU-only mode."
    echo "For GPU support, please install NVIDIA drivers and CUDA toolkit."
    echo ""
}

# Check if GPU is actually available
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    echo "[OK] GPU detected: $GPU_NAME"
    echo ""
    echo "================================================"
    echo "[OK] GPU Installation Complete!"
    echo "================================================"
else
    echo ""
    echo "================================================"
    echo "[WARNING] Installation Complete (CPU Mode)"
    echo "================================================"
fi

echo ""
echo "[INFO] IMPORTANT USAGE INSTRUCTIONS:"
echo ""
echo "To preserve GPU support, you MUST activate the virtual"
echo "environment before running commands:"
echo ""
echo "  source .venv/bin/activate"
echo "  python -m expert_among_us --help"
echo ""
echo "[ERROR] DO NOT use 'uv run' as it will revert to CPU-only PyTorch"
echo ""
echo "To verify GPU is working:"
echo "  source .venv/bin/activate"
echo "  python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\""
echo ""
echo "[INFO] For more information, see README.md"
echo ""