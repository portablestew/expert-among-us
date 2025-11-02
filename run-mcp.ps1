$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Auto-install if venv doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Virtual environment not found. Running install-gpu.ps1..."
    & .\install-gpu.ps1
}

& .venv\Scripts\python.exe -m expert_among_us.__mcp__ $args