param([switch]$OpenBrowser)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$workspaceUrl = 'http://127.0.0.1:8000/'
$workspaceMarker = 'data-veritas-workspace="current"'

if ($OpenBrowser) {
    try {
        $response = Invoke-WebRequest -Uri $workspaceUrl -UseBasicParsing -TimeoutSec 2
    } catch {
        $response = $null
    }
    if ($response -and $response.Content.Contains($workspaceMarker)) {
        Start-Process -FilePath $workspaceUrl
        return
    }
}

$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $projectPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$env:HF_HOME = Join-Path $PSScriptRoot 'data/cache'
$browserJob = $null
if ($OpenBrowser) {
    $browserJob = Start-Job -ArgumentList $workspaceUrl, $workspaceMarker -ScriptBlock {
        param($url, $marker)
        $deadline = [DateTime]::UtcNow.AddSeconds(45)
        do {
            try {
                $page = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
                if ($page.Content.Contains($marker)) {
                    Start-Process -FilePath $url
                    return
                }
            } catch {
                # The local server may still be starting.
            }
            Start-Sleep -Milliseconds 500
        } while ([DateTime]::UtcNow -lt $deadline)
    }
}

try {
    Write-Host "Veritas workspace: $workspaceUrl"
    & $projectPython -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
    $serverExit = $LASTEXITCODE
} finally {
    if ($browserJob) { Remove-Job -Job $browserJob -Force }
}
exit $serverExit
