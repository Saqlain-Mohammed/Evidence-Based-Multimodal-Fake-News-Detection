# Resume interrupted downloads, then train and evaluate each independent detector.
# No training data, uploads or account history is published by this script.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
function Invoke-TrainingStep {
    param([string[]]$StepArguments)
    & $projectPython @StepArguments
    if ($LASTEXITCODE -ne 0) { throw "Training step failed: $($StepArguments[0])" }
}
Invoke-TrainingStep @('training/ranged_download.py', 'uadfv')
Invoke-TrainingStep @('training/import_uadfv.py')
Invoke-TrainingStep @('training/train_media.py', '--task', 'video', '--data', 'data/prepared/uadfv', '--dataset-name', 'UADFV historical paired-source baseline', '--epochs', '15', '--batch', '8', '--evaluation-scope', 'small-internal-source-split', '--limitations', 'Only 98 historical clips. Source-video-disjoint internal split; inserted identities can overlap. This is not modern or external deepfake accuracy. No tamper timestamp validation.')
Copy-Item -LiteralPath data/public/uadfv/provenance.json -Destination models/public/video/dataset_provenance.json
Invoke-TrainingStep @('training/ranged_download.py', 'casia')
Invoke-TrainingStep @('training/import_public_media.py', '--task', 'casia')
Invoke-TrainingStep @('training/train_media.py', '--task', 'image_manipulation', '--data', 'data/prepared/casia', '--dataset-name', 'CASIA v2 internal source-connected split', '--epochs', '15', '--batch', '64', '--evaluation-scope', 'internal-source-connected-split')
Copy-Item -LiteralPath data/public/casia/provenance.json -Destination models/public/image_manipulation/dataset_provenance.json
Invoke-TrainingStep @('training/ranged_download.py', 'asvspoof')
Invoke-TrainingStep @('training/import_public_media.py', '--task', 'audio')
Invoke-TrainingStep @('training/train_audio_official.py')
Invoke-TrainingStep @('training/train_mask.py')
Write-Output 'Evaluation reports are in models/public.'
