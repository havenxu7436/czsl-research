param([string]$Server = 'xhw@211.87.232.122')
$ErrorActionPreference = 'Stop'
$g0Server = $Server
$g0Package = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'trustg0_three_v1_10ep.zip'
if (!(Test-Path -LiteralPath $g0Package)) { throw "Package not found: $g0Package" }
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $g0Server 'mkdir -p /var/tmp/xhw'
if ($LASTEXITCODE -ne 0) { throw 'Remote directory creation failed' }
scp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $g0Package "$g0Package.sha256" "${g0Server}:/var/tmp/xhw/"
if ($LASTEXITCODE -ne 0) { throw 'Upload failed; rerun upload.ps1' }
Write-Host 'UPLOAD_COMPLETE: /var/tmp/xhw/trustg0_three_v1_10ep.zip'
