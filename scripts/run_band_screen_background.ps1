[CmdletBinding()]
param(
    [string]$Folds = "0,1,2",
    [string]$Json = "results/band_screen_soft_folds012.json",
    [string]$LogFile = "results/band_screen_soft.log",
    [switch]$NoKeepAwake
)

# Launch the band-condition screen as a detached background process so it
# survives closing the terminal, VS Code, or the screen turning off.
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$python = (Get-Command python -ErrorAction Stop).Source

$resultsDir = Join-Path $repoRoot "results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
$pidFile = Join-Path $resultsDir "band_screen.pid"
$guardPidFile = Join-Path $resultsDir "band_screen_keepawake.pid"

# Refuse to start twice: two screens would each filter the full bank.
if (Test-Path $pidFile) {
    $existingPid = [int](Get-Content $pidFile -ErrorAction Stop | Select-Object -First 1)
    $existing = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($existing) {
        throw "Band screen is already running (PID $existingPid)."
    }
    Remove-Item $pidFile -Force
}

$battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($battery -and $battery.BatteryStatus -ne 2) {
    Write-Warning "The laptop is on battery (status $($battery.BatteryStatus)). Plug in the charger."
}

# Hidden guard that blocks idle sleep (the display may still switch off).
if (-not $NoKeepAwake) {
    $guardScript = Join-Path $repoRoot "scripts\keep_awake_guard.ps1"
    $guard = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $guardScript) `
        -WindowStyle Hidden -PassThru
    $guard.Id | Set-Content -Path $guardPidFile -Encoding ascii
    Write-Host "Keep-awake guard started (PID $($guard.Id))."
}

$stdout = $LogFile
if (-not [System.IO.Path]::IsPathRooted($stdout)) {
    $stdout = Join-Path $repoRoot $stdout
}
$stdoutDir = Split-Path -Parent $stdout
New-Item -ItemType Directory -Force -Path $stdoutDir | Out-Null
$stderr = Join-Path $resultsDir "band_screen_stderr.log"
$arguments = @(
    "-u",
    "scripts/screen_band_conditions.py",
    "--folds", $Folds,
    "--json", $Json
)

$proc = Start-Process -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

$proc.Id | Set-Content -Path $pidFile -Encoding ascii
# Remember which log this run writes, so check_band_screen.ps1 finds it.
$stdout | Set-Content -Path (Join-Path $resultsDir "band_screen.logpath") -Encoding ascii
Start-Sleep -Seconds 3
$proc.Refresh()
if ($proc.HasExited) {
    throw "Band screen exited immediately (code $($proc.ExitCode)). Check $stderr"
}

Write-Host ""
Write-Host "Band screen started in the background."
Write-Host "  PID    : $($proc.Id)  (saved to $pidFile)"
Write-Host "  Folds  : $Folds"
Write-Host "  JSON   : $Json"
Write-Host "  Log    : $LogFile"
Write-Host "  Stdout : $stdout"
Write-Host "  Stderr : $stderr"
Write-Host ""
Write-Host "Check progress:"
Write-Host "  Get-Content $LogFile -Tail 20 -Wait"
Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_band_screen.ps1"
Write-Host ""
Write-Host "IMPORTANT: keep the charger connected and leave the lid OPEN."
Write-Host "The guard prevents idle sleep; closing the lid still sleeps the machine."
