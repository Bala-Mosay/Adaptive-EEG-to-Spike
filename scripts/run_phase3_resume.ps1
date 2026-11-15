[CmdletBinding()]
param(
    [string]$VenvDir = ".venv",
    [string]$Config = "config/phase3_config.yaml",
    [string]$LogFile = "results/phase3_c1_run.log",
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot $VenvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found. Run scripts\setup_windows.ps1 first."
}

$checkpoint = Join-Path $repoRoot "results\phase3_c1\checkpoint.json"
if (-not $Fresh -and (Test-Path $checkpoint)) {
    Write-Host "=== Existing checkpoint ==="
    Get-Content $checkpoint
    Write-Host ""
}

$arguments = @(
    "-u",
    "experiments/run_phase3.py",
    "--config", $Config,
    "--log-file", $LogFile
)
if ($Fresh) {
    $arguments += "--overwrite"
} else {
    $arguments += "--resume"
}

Write-Host "=== Starting Phase 3 C1 (Ctrl+C is safe; progress is checkpointed per fold) ==="
Write-Host "Log file: $LogFile"
Write-Host ""

& $venvPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Run failed with exit code $LASTEXITCODE"
}
