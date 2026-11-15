[CmdletBinding()]
param(
    [string]$OutFile = "eeg_preprocessed_103.zip"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$cacheDir = Join-Path $repoRoot "outputs\preprocessed"
$files = Get-ChildItem $cacheDir -Filter "S*.npz" -ErrorAction SilentlyContinue
if (-not $files -or $files.Count -eq 0) {
    throw "No preprocessed caches found in $cacheDir"
}

$outPath = Join-Path $repoRoot $OutFile
Write-Host "Packing $($files.Count) cache files into $outPath ..."
Write-Host "This is ~2.3 GB and can take a few minutes."

Compress-Archive -Path $files.FullName -DestinationPath $outPath -CompressionLevel Optimal -Force

$sizeGb = (Get-Item $outPath).Length / 1GB
Write-Host ("Wrote {0} ({1:N2} GB)" -f $OutFile, $sizeGb)
Write-Host "Copy this file to the other machine and extract it into outputs\preprocessed."
