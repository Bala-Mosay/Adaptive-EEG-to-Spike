$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pidFile = Join-Path $repoRoot "results\phase3_c1.pid"
$guardPidFile = Join-Path $repoRoot "results\phase3_c1_keepawake.pid"

if (Test-Path $pidFile) {
    $targetPid = [int](Get-Content $pidFile | Select-Object -First 1)
    $proc = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "Stopping Phase 3 process PID $targetPid ..."
        Stop-Process -Id $targetPid -Force
        Write-Host "Stopped. Completed folds remain in results\phase3_c1\."
    }
    else {
        Write-Host "Phase 3 process is not running."
    }
    Remove-Item $pidFile -Force
}
else {
    Write-Host "No Phase 3 PID file."
}

if (Test-Path $guardPidFile) {
    $guardPid = [int](Get-Content $guardPidFile | Select-Object -First 1)
    $guard = Get-Process -Id $guardPid -ErrorAction SilentlyContinue
    if ($guard) {
        Write-Host "Stopping keep-awake guard PID $guardPid ..."
        Stop-Process -Id $guardPid -Force
    }
    Remove-Item $guardPidFile -Force
}
