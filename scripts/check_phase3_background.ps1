$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pidFile = Join-Path $repoRoot "results\phase3_c1.pid"
$guardPidFile = Join-Path $repoRoot "results\phase3_c1_keepawake.pid"

Write-Host "=== Process ==="
if (Test-Path $pidFile) {
    $targetPid = [int](Get-Content $pidFile | Select-Object -First 1)
    $proc = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "RUNNING  PID=$targetPid  started=$($proc.StartTime)  CPU=$([math]::Round($proc.CPU, 1))s"
    }
    else {
        Write-Host "NOT RUNNING (stale PID $targetPid)"
    }
}
else {
    Write-Host "NOT RUNNING (no PID file)"
}

if (Test-Path $guardPidFile) {
    $guardPid = [int](Get-Content $guardPidFile | Select-Object -First 1)
    $guard = Get-Process -Id $guardPid -ErrorAction SilentlyContinue
    if ($guard) {
        Write-Host "Keep-awake guard: RUNNING PID=$guardPid"
    }
    else {
        Write-Host "Keep-awake guard: NOT RUNNING"
    }
}

Write-Host ""
Write-Host "=== Checkpoint ==="
$checkpoint = Join-Path $repoRoot "results\phase3_c1\checkpoint.json"
if (Test-Path $checkpoint) {
    Get-Content $checkpoint
}
else {
    Write-Host "No checkpoint yet."
}

Write-Host ""
Write-Host "=== Log tail ==="
$log = Join-Path $repoRoot "results\phase3_c1_run.log"
if (Test-Path $log) {
    Get-Content $log -Tail 20
}
else {
    Write-Host "No log file yet."
}

Write-Host ""
Write-Host "=== Stderr tail ==="
$stderr = Join-Path $repoRoot "results\phase3_c1_stderr.log"
if (Test-Path $stderr) {
    Get-Content $stderr -Tail 20
}
else {
    Write-Host "No stderr file yet."
}
