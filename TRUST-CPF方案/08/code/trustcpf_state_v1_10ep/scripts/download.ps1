param(
    [string]$Server = "xhw@211.87.232.122",
    [string]$Destination = "C:\Users\llenovo\Documents\ChatGPT\USG\output\state_results"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$cpfArchive = Join-Path $Destination "trustcpf_state_v1_10ep_results.tar.gz"
$cpfHashFile = "$cpfArchive.sha256"
scp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$($Server):/data/xhw/experiments/trustcpf_state_v1_10ep_results.tar.gz" $cpfArchive
if ($LASTEXITCODE -ne 0) { throw "Download interrupted; rerun this command" }
scp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$($Server):/data/xhw/experiments/trustcpf_state_v1_10ep_results.tar.gz.sha256" $cpfHashFile
if ($LASTEXITCODE -ne 0) { throw "Checksum download failed" }
$cpfExpected = ((Get-Content -LiteralPath $cpfHashFile -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
$cpfActual = (Get-FileHash -LiteralPath $cpfArchive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($cpfActual -ne $cpfExpected) { throw "SHA256 mismatch; download again before extracting" }
$cpfExtract = Join-Path $Destination ("server_results_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $cpfExtract | Out-Null
tar -xzf $cpfArchive -C $cpfExtract
if ($LASTEXITCODE -ne 0) { throw "Extraction failed" }
Write-Host "SAVED: $cpfExtract\trustcpf_state_v1_10ep\work"
Write-Host "Main files: summary.csv, paired.csv, aggregates.json, q5_decisions.csv, training_curves.csv"

