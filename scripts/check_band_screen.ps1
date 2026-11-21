[CmdletBinding()]
param([string]$LogFile = "results/band_screen_soft.log")

# Quick health check for the detached band screen.
$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pidFile = Join-Path $repoRoot "results/band_screen.pid"
if (-not (Test-Path $pidFile)) {
    Write-Host "No PID file found - the screen has not been started."
    return
}

$procId = [int](Get-Content $pidFile | Select-Object -First 1)
$proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
if ($proc) {
    $rss = [math]::Round($proc.WorkingSet64 / 1GB, 2)
    $cpu = [math]::Round($proc.CPU, 1)
    Write-Host "RUNNING  pid=$procId  cpu=${cpu}s  rss=${rss}GB"
} else {
    Write-Host "STOPPED  pid=$procId is not running (finished, crashed, or was killed)."
    $err = Join-Path $repoRoot "results/band_screen_stderr.log"
    if (Test-Path $err) {
        Write-Host ""
        Write-Host "--- stderr (last 15 lines) ---"
        Get-Content $err -Tail 15
    }
}

Write-Host ""
Write-Host "--- log (last 25 lines) ---"
$logPathFile = Join-Path $repoRoot "results/band_screen.logpath"
if (Test-Path $logPathFile) {
    $recorded = (Get-Content $logPathFile | Select-Object -First 1).Trim()
    if ($recorded) { $LogFile = $recorded }
}
if (-not (Test-Path $LogFile)) {
    # The first launcher build advertised this name but wrote stdout.log.
    $fallback = Join-Path $repoRoot "results/band_screen_stdout.log"
    if (Test-Path $fallback) {
        $LogFile = $fallback
    }
}
if (Test-Path $LogFile) {
    Get-Content $LogFile -Tail 25
} else {
    Write-Host "(no log at $LogFile yet)"
}
