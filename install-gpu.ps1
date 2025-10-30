# Expert Among Us - GPU Installation Script (PowerShell)
# Requires PowerShell 5.1 or higher

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Expert Among Us - GPU Installation" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = & python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    
    $versionMatch = $pythonVersion -match "Python (\d+\.\d+)"
    if (-not $versionMatch) {
        throw "Could not parse Python version"
    }
    
    $version = [version]$matches[1]
    $requiredVersion = [version]"3.12"
    
    if ($version -lt $requiredVersion) {
        Write-Host "[ERROR] Error: Python $version found, but Python 3.12+ is required" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "[OK] Python $version detected" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Error: Python 3 is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.12 or higher from https://www.python.org/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Check for uv
Write-Host "Checking for uv package manager..." -ForegroundColor Yellow
try {
    $uvVersion = & uv --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] uv is already installed" -ForegroundColor Green
    } else {
        throw "uv not found"
    }
} catch {
    Write-Host "[INSTALL] Installing uv package manager..." -ForegroundColor Yellow
    
    # Install uv using the official installer
    try {
        Invoke-RestMethod -Uri "https://astral.sh/uv/install.ps1" | Invoke-Expression
        
        # Refresh PATH for current session
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        # Verify installation
        $uvVersion = & uv --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "uv installation verification failed"
        }
        Write-Host "[OK] uv installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Error: uv installation failed" -ForegroundColor Red
        Write-Host "Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "Installing base project dependencies..." -ForegroundColor Yellow
try {
    & uv sync
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed"
    }
} catch {
    Write-Host "[ERROR] Error: Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Activating virtual environment and installing GPU-enabled PyTorch..." -ForegroundColor Yellow
Write-Host ""

# Activate virtual environment
$activateScript = ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "[ERROR] Error: Virtual environment not found at .venv" -ForegroundColor Red
    exit 1
}

# Run activation and PyTorch installation in the same context
try {
    Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Yellow
    & uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch installation failed"
    }
} catch {
    Write-Host "[ERROR] Error: Failed to install GPU-enabled PyTorch" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Verifying GPU detection..." -ForegroundColor Yellow

# Check CUDA availability
$cudaCheck = & .venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())" 2>&1
$cudaAvailable = $cudaCheck -eq "True"

if (-not $cudaAvailable) {
    Write-Host ""
    Write-Host "[WARNING] WARNING: CUDA is not available" -ForegroundColor Yellow
    Write-Host "This could mean:" -ForegroundColor Yellow
    Write-Host "  1. You don't have an NVIDIA GPU" -ForegroundColor Yellow
    Write-Host "  2. NVIDIA drivers are not installed" -ForegroundColor Yellow
    Write-Host "  3. CUDA toolkit is not properly configured" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To check your GPU and drivers:" -ForegroundColor Yellow
    Write-Host "  nvidia-smi" -ForegroundColor White
    Write-Host ""
    Write-Host "The installation will continue, but you'll be using CPU-only mode." -ForegroundColor Yellow
    Write-Host "For GPU support, please install NVIDIA drivers and CUDA toolkit." -ForegroundColor Yellow
    Write-Host ""
}

if ($cudaAvailable) {
    $gpuName = & .venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
    Write-Host "[OK] GPU detected: $gpuName" -ForegroundColor Green
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "[OK] GPU Installation Complete!" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "[WARNING] Installation Complete (CPU Mode)" -ForegroundColor Yellow
    Write-Host "================================================" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "[INFO] IMPORTANT USAGE INSTRUCTIONS:" -ForegroundColor Cyan
Write-Host ""
Write-Host "To preserve GPU support, you MUST activate the virtual" -ForegroundColor Yellow
Write-Host "environment before running commands:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python -m expert_among_us --help" -ForegroundColor White
Write-Host ""
Write-Host "[ERROR] DO NOT use 'uv run' as it will revert to CPU-only PyTorch" -ForegroundColor Red
Write-Host ""
Write-Host "To verify GPU is working:" -ForegroundColor Yellow
Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python -c `"import torch; print(f'CUDA available: {torch.cuda.is_available()}')`"" -ForegroundColor White
Write-Host ""
Write-Host "[INFO] For more information, see README.md" -ForegroundColor Cyan
Write-Host ""