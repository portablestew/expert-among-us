# Expert Among Us - Run Script (PowerShell)
# This script runs the application using the virtual environment
# without leaving the venv activated after the script terminates

$ErrorActionPreference = "Stop"

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Error: Virtual environment not found" -ForegroundColor Red
    Write-Host "Please run the appropriate install script first:" -ForegroundColor Yellow
    Write-Host "  .\install-cpu.ps1    (for CPU-only installation)" -ForegroundColor White
    Write-Host "  .\install-gpu.ps1    (for GPU installation)" -ForegroundColor White
    exit 1
}

# Run the application using the venv's python directly
# This avoids activating the venv, preventing any side effects
& .venv\Scripts\python.exe -m expert_among_us $args
exit $LASTEXITCODE