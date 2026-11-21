$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    python -m venv .venv
    & $projectPython -m pip install -r requirements.txt
}
$env:HF_HOME = Join-Path $PSScriptRoot 'data/cache'
& $projectPython -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
