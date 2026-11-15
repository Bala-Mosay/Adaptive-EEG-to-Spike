[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$VenvDir = ".venv",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "=== Adaptive EEG-to-Spike: Phase 3 setup ==="
Write-Host "Repo root: $repoRoot"

$version = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) {
    throw "Python was not found. Install Python 3.10+ and re-run with -Python <path-to-python.exe>"
}
if ([version]$version -lt [version]"3.10") {
    throw "Python 3.10+ is required; found $version"
}
Write-Host "Using Python $version"

$venvPath = Join-Path $repoRoot $VenvDir
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment: $venvPath"
    & $Python -m venv $venvPath
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment python not found at $venvPython"
}

Write-Host "Installing dependencies (first install can take several minutes)..."
& $venvPython -m pip install --upgrade pip wheel setuptools
& $venvPython -m pip install -r (Join-Path $repoRoot "requirements.txt")
& $venvPython -m pip install pytest

Write-Host ""
Write-Host "=== Verifying environment and data ==="
& $venvPython (Join-Path $repoRoot "scripts\verify_setup.py")
if ($LASTEXITCODE -ne 0) {
    throw "Setup verification failed. Fix the messages above before running the experiment."
}

if (-not $SkipTests) {
    Write-Host ""
    Write-Host "=== Running test suite ==="
    & $venvPython -m pytest tests/ -q
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Start or resume the run with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_phase3_resume.ps1"
