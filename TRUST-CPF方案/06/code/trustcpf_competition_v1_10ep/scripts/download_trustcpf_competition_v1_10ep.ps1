param(
    [string]$LocalRoot = 'C:\Users\llenovo\Documents\ChatGPT\USG',
    [string]$PythonPath = 'C:\Users\llenovo\anaconda3\envs\ai-lab\python.exe',
    [string]$Server = 'xhw@211.87.232.122'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$cpfRemoteManifest = '/data/xhw/experiments/trustcpf_competition_v1_10ep/download.json'
$cpfOptions = @('-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6')
$cpfExtractor = Join-Path $PSScriptRoot 'extract_results.py'
if (-not (Test-Path -LiteralPath $cpfExtractor -PathType Leaf)) {
    $cpfExtractor = Join-Path $LocalRoot 'trustcpf_competition_v1_10ep\scripts\extract_results.py'
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw "Python not found: $PythonPath" }
if (-not (Test-Path -LiteralPath $cpfExtractor -PathType Leaf)) { throw "Extractor not found: $cpfExtractor" }
$null = New-Item -ItemType Directory -Force -Path $LocalRoot
$cpfLocalManifest = Join-Path $LocalRoot 'trustcpf_competition_v1_10ep_download.json'
function Receive-CpfFile {
    param([string]$RemoteFile,[string]$LocalFile)
    if ([string]::IsNullOrWhiteSpace($RemoteFile)) { throw 'Empty remote path.' }
    for ($cpfAttempt = 1; $cpfAttempt -le 3; $cpfAttempt++) {
        Write-Host "Downloading (attempt $cpfAttempt): $RemoteFile"
        & scp @cpfOptions "${Server}:$RemoteFile" $LocalFile
        $cpfTransferExit = $LASTEXITCODE
        if ($cpfTransferExit -eq 0 -and (Test-Path -LiteralPath $LocalFile -PathType Leaf)) { return }
    }
    throw "Download failed: $RemoteFile"
}
Receive-CpfFile -RemoteFile $cpfRemoteManifest -LocalFile $cpfLocalManifest
$cpfDescriptor = Get-Content -LiteralPath $cpfLocalManifest -Raw | ConvertFrom-Json
$cpfName = [string]$cpfDescriptor.name
if ($cpfName -notmatch '^trustcpf_competition_v1_10ep_results_[0-9]{8}_[0-9]{6}\.tar\.gz$') { throw 'Invalid archive name in download.json.' }
$cpfRemoteArchive = "/data/xhw/experiments/$cpfName"
if ([string]$cpfDescriptor.archive -cne $cpfRemoteArchive) { throw 'Archive path does not match the experiment directory.' }
$cpfExpected = [string]$cpfDescriptor.sha256
if ($cpfExpected -notmatch '^[a-fA-F0-9]{64}$') { throw 'Invalid SHA256 in download.json.' }
$cpfArchive = Join-Path $LocalRoot $cpfName
$cpfPartial = "$cpfArchive.part"
$cpfAlreadyHave = $false
if (Test-Path -LiteralPath $cpfArchive -PathType Leaf) {
    $cpfAlreadyHave = (Get-FileHash -LiteralPath $cpfArchive -Algorithm SHA256).Hash -ieq $cpfExpected
}
if (-not $cpfAlreadyHave) {
    Receive-CpfFile -RemoteFile $cpfRemoteArchive -LocalFile $cpfPartial
    if ((Get-Item -LiteralPath $cpfPartial).Length -ne [long]$cpfDescriptor.bytes) { throw 'Downloaded file size differs.' }
    Write-Host 'Checking SHA256...'
    if ((Get-FileHash -LiteralPath $cpfPartial -Algorithm SHA256).Hash -ine $cpfExpected) { throw 'SHA256 mismatch. Rerun this script.' }
    Move-Item -LiteralPath $cpfPartial -Destination $cpfArchive -Force
}
$cpfExtract = Join-Path $LocalRoot ($cpfName -replace '\.tar\.gz$','')
& $PythonPath $cpfExtractor $cpfArchive $cpfExtract
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed.' }
$cpfWork = Join-Path $cpfExtract 'trustcpf_competition_v1_10ep\work'
$cpfStatus = Get-Content -LiteralPath (Join-Path $cpfWork 'status.json') -Raw | ConvertFrom-Json
if (-not $cpfStatus.complete) { throw "Archived experiment incomplete: $cpfWork" }
Write-Host 'DOWNLOAD_SHA256_EXTRACTION_COMPLETE' -ForegroundColor Green
Write-Host "Results: $cpfWork"
Write-Host "Models: $($cpfStatus.metrics)/15; Q5 decision audits: $($cpfStatus.utility_audits)/3; complete stages: $($cpfStatus.completed_stages)/42"
