# scripts/activate.ps1
# Helper to activate the project virtual environment on Windows.
# Usage: . .\scripts\activate.ps1   (dot-source to affect current shell)

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"

if (-Not (Test-Path $VenvActivate)) {
    Write-Host "Virtual environment not found. Creating one now..." -ForegroundColor Yellow
    python -m venv "$ProjectRoot\.venv"
    Write-Host "Created .venv. Installing dependencies..." -ForegroundColor Yellow
    & "$ProjectRoot\.venv\Scripts\pip.exe" install -e "$ProjectRoot[dev]" --quiet
}

Write-Host "Activating stock-agent virtual environment..." -ForegroundColor Cyan
. $VenvActivate
Write-Host "Done. Python: $(python --version)" -ForegroundColor Green
