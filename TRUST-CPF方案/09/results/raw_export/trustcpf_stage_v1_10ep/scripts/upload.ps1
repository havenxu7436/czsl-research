param(
    [string]$Server = "xhw@211.87.232.122",
    [string]$LocalZip = "C:\Users\llenovo\Documents\ChatGPT\USG\output\trustcpf_stage_v1_10ep.zip"
)
$ErrorActionPreference = "Stop"
if (!(Test-Path -LiteralPath $LocalZip)) { throw "Missing archive: $LocalZip" }
$cpfHashFile = "$LocalZip.sha256"
$cpfExpected = ((Get-Content -LiteralPath $cpfHashFile -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
$cpfActual = (Get-FileHash -LiteralPath $LocalZip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($cpfActual -ne $cpfExpected) { throw "Local ZIP SHA256 mismatch" }
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $Server "mkdir -p /data/xhw/experiments"
if ($LASTEXITCODE -ne 0) { throw "Remote directory creation failed" }
scp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $LocalZip $cpfHashFile "$($Server):/data/xhw/experiments/"
if ($LASTEXITCODE -ne 0) { throw "Upload interrupted; rerun this command" }
Write-Host "Upload complete. Continue with the server commands."

