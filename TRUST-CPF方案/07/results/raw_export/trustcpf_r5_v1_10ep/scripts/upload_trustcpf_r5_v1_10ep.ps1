param(
    [string]$BundleRoot = 'C:\Users\llenovo\Documents\ChatGPT\USG\output',
    [string]$Server = 'xhw@211.87.232.122'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$cpfZip = Join-Path $BundleRoot 'trustcpf_r5_v1_10ep.zip'
$cpfChecksum = "$cpfZip.sha256"
$cpfOptions = @('-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6')
$cpfExpected = ((Get-Content -LiteralPath $cpfChecksum -Raw).Trim() -split '\s+')[0]
if ((Get-FileHash -LiteralPath $cpfZip -Algorithm SHA256).Hash -ine $cpfExpected) { throw 'Local ZIP checksum mismatch.' }
& ssh @cpfOptions $Server 'mkdir -p /data/xhw/experiments'
if ($LASTEXITCODE -ne 0) { throw 'Cannot create the remote experiment directory.' }
& scp @cpfOptions $cpfZip $cpfChecksum "${Server}:/data/xhw/experiments/"
if ($LASTEXITCODE -ne 0) { throw 'Upload failed. Rerun this script.' }
& ssh @cpfOptions $Server 'cd /data/xhw/experiments && sha256sum -c trustcpf_r5_v1_10ep.zip.sha256 && unzip -oq trustcpf_r5_v1_10ep.zip -d .'
if ($LASTEXITCODE -ne 0) { throw 'Remote verification or extraction failed.' }
Write-Host 'UPLOAD_AND_EXTRACTION_COMPLETE: /data/xhw/experiments/trustcpf_r5_v1_10ep' -ForegroundColor Green
