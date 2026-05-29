# Expert Among Us - Local Development Setup (PowerShell)
# For end-users: install with `uv tool install expert-among-us` — see README.

$ErrorActionPreference = "Stop"

Write-Host "Expert Among Us - Local Development Setup" -ForegroundColor Cyan
Write-Host ""

# Check for uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..." -ForegroundColor Yellow
    Invoke-RestMethod -Uri "https://astral.sh/uv/install.ps1" | Invoke-Expression
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
}

Write-Host "Syncing dependencies (CUDA-enabled PyTorch)..." -ForegroundColor Yellow
uv sync

Write-Host ""
Write-Host "Verifying install..." -ForegroundColor Yellow
uv run expert-among-us --help

Write-Host ""
Write-Host "Done. Use 'uv run expert-among-us' to run from this clone." -ForegroundColor Green
