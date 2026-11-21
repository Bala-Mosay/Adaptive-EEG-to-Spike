[CmdletBinding()]
param(
    [string]$VenvDir = ".venv",
    [string]$Config = "config/phase3_config.yaml",
    [string]$LogFile = "results/phase3_c1_run.log",
    [switch]$Fresh,
    [switch]$NoKeepAwake
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot "$VenvDir\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if (-not $systemPython) {
        throw "No virtual environment and no 'python' on PATH. Run scripts\setup_windows.ps1 first."
    }
    $venvPython = $systemPython.Source
    Write-Warning "No .venv found; using system Python: $venvPython"
}

$resultsDir = Join-Path $repoRoot "results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
$pidFile = Join-Path $resultsDir "phase3_c1.pid"
$guardPidFile = Join-Path $resultsDir "phase3_c1_keepawake.pid"

# Refuse to start twice.
if (Test-Path $pidFile) {
    $existingPid = [int](Get-Content $pidFile -ErrorAction Stop | Select-Object -First 1)
    $existing = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($existing) {
        throw "Phase 3 is already running (PID $existingPid). Use scripts\stop_phase3_background.ps1 first."
    }
    Remove-Item $pidFile -Force
}

$battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($battery -and $battery.BatteryStatus -ne 2) {
    Write-Warning "The laptop is currently on battery (status $($battery.BatteryStatus))."
    Write-Warning "Plug in the charger before leaving, or the process will stop when the battery dies."
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

# Start a hidden guard that prevents idle sleep (screen may still turn off).
if (-not $NoKeepAwake) {
    $guardScript = Join-Path $repoRoot "scripts\keep_awake_guard.ps1"
    $guard = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $guardScript) `
        -WindowStyle Hidden -PassThru
    $guard.Id | Set-Content -Path $guardPidFile -Encoding ascii
    Write-Host "Keep-awake guard started (PID $($guard.Id))."
}

$stdout = Join-Path $resultsDir "phase3_c1_stdout.log"
$stderr = Join-Path $resultsDir "phase3_c1_stderr.log"
$proc = Start-Process -FilePath $venvPython `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

$proc.Id | Set-Content -Path $pidFile -Encoding ascii
Start-Sleep -Seconds 2
$proc.Refresh()
if ($proc.HasExited) {
    throw "Phase 3 exited immediately (code $($proc.ExitCode)). Check $stderr"
}

Write-Host ""
Write-Host "Phase 3 C1 started in the background."
Write-Host "  PID      : $($proc.Id)"
Write-Host "  Log      : $LogFile"
Write-Host "  Stdout   : $stdout"
Write-Host "  Stderr   : $stderr"
Write-Host "  Check    : powershell -ExecutionPolicy Bypass -File scripts\check_phase3_background.ps1"
Write-Host "  Stop     : powershell -ExecutionPolicy Bypass -File scripts\stop_phase3_background.ps1"
Write-Host ""
Write-Host "IMPORTANT: keep the charger connected and leave the lid open."
Write-Host "The keep-awake guard prevents idle sleep, but closing the lid still"
Write-Host "uses the Windows lid-close action (usually sleep)."
