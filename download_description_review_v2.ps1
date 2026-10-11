param(
    [string]$DestinationDirectory = (Join-Path $PSScriptRoot 'output\semantic_review_v2')
)

$ErrorActionPreference = 'Stop'
$serverAddress = 'xhw@211.87.232.122'
$expectedRemotePrefix = '/data/xhw/czsl_repro/experiments/ekr0601_transition_v1/diagnostics/semantic_gate_audit_v2_'
$connectionOptions = @('-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1', '-o', 'StrictHostKeyChecking=yes')

# This command only reads the newest v2 review and its checksum on the server.
$remoteReadCommand = @'
/data/xhw/miniconda3/envs/cpf/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('/data/xhw/czsl_repro/experiments/ekr0601_transition_v1/diagnostics')
files = sorted(root.glob('semantic_gate_audit_v2_*/description_review.md'))
if not files:
    raise SystemExit('No v2 description_review.md found')
path = files[-1]
print(json.dumps({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
PY
'@

Write-Host 'Step 1: locate the newest v2 review. Enter the SSH password if requested.'
$metadataOutput = @(& ssh @connectionOptions -T $serverAddress $remoteReadCommand)
if ($LASTEXITCODE -ne 0) {
    throw 'SSH failed. No file was downloaded.'
}
$metadataLines = @($metadataOutput | Where-Object { $_ -match '^\s*\{.*"path"\s*:' })
if ($metadataLines.Count -ne 1) {
    throw 'Unexpected server output. Expected one review metadata record.'
}
$metadata = $metadataLines[0] | ConvertFrom-Json
$remotePath = [string]$metadata.path
if (-not $remotePath.StartsWith($expectedRemotePrefix) -or
    $remotePath -notmatch '/semantic_gate_audit_v2_[0-9]{8}_[0-9]{6}/description_review\.md$' -or
    [string]$metadata.sha256 -notmatch '^[0-9a-f]{64}$') {
    throw 'Unexpected remote review path or checksum.'
}

$downloadDirectory = [System.IO.Path]::GetFullPath($DestinationDirectory)
New-Item -ItemType Directory -Path $downloadDirectory -Force | Out-Null
$localFilename = Join-Path $downloadDirectory 'description_review.md'
$partialFilename = Join-Path $downloadDirectory ('description_review.download_' + [guid]::NewGuid().ToString('N') + '.md')

Write-Host ('REMOTE_FILE=' + $remotePath)
Write-Host 'Step 2: download the review. SSH may request the password again.'
& scp @connectionOptions "${serverAddress}:$remotePath" $partialFilename
if ($LASTEXITCODE -ne 0) {
    throw ('SCP failed. Any partial download is retained at: ' + $partialFilename)
}
$actualDigest = (Get-FileHash -LiteralPath $partialFilename -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualDigest -ne [string]$metadata.sha256) {
    throw ('Checksum mismatch. The existing review is untouched. Download retained at: ' + $partialFilename)
}

if (Test-Path -LiteralPath $localFilename) {
    $oldDigest = (Get-FileHash -LiteralPath $localFilename -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($oldDigest -ne $actualDigest) {
        $savedFilename = Join-Path $downloadDirectory ('description_review.previous_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8) + '.md')
        # Both literal paths stay inside the verified local download directory.
        Move-Item -LiteralPath $localFilename -Destination $savedFilename
        Write-Host ('PREVIOUS_REVIEW_SAVED=' + $savedFilename)
    }
}
Move-Item -LiteralPath $partialFilename -Destination $localFilename -Force
Write-Host 'SHA256 CHECK PASSED'
Write-Host ('SAVED_FILE=' + $localFilename)
Write-Host ('SHA256=' + $actualDigest)
