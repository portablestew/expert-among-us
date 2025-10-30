# Expert Among Us - CPU Installation Script (PowerShell)
# Requires PowerShell 5.1 or higher

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Expert Among Us - CPU Installation" -ForegroundColor Cyan
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
Write-Host "Installing project dependencies..." -ForegroundColor Yellow
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
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "[OK] Installation Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the application, use:" -ForegroundColor Yellow
Write-Host "  uv run python -m expert_among_us --help" -ForegroundColor White
Write-Host ""
Write-Host "Or activate the virtual environment:" -ForegroundColor Yellow
Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python -m expert_among_us --help" -ForegroundColor White
Write-Host ""
Write-Host "[INFO] For more information, see README.md" -ForegroundColor Cyan
Write-Host ""