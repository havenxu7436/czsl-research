param(
    [Parameter(Mandatory = $true)][string]$RemoteExport,
    [string]$SshTarget = 'xhw@211.87.232.122',
    [int]$Port = 22,
    [string]$LocalRoot = 'C:\Users\llenovo\Desktop\CPF_all_results',
    [switch]$DownloadOnly
)
$ErrorActionPreference = 'Stop'
if ($RemoteExport -notmatch '^/[A-Za-z0-9_./-]+$' -or $RemoteExport -match '(^|/)\.\.(/|$)') {
    throw 'RemoteExport must be the absolute ASCII path printed as REMOTE_EXPORT.'
}
if ($SshTarget -notmatch '^[A-Za-z0-9_.@-]+$') { throw 'Invalid SSH target.' }
if ($Port -lt 1 -or $Port -gt 65535) { throw 'Invalid SSH port.' }
foreach ($taskProgram in @('scp', 'tar')) {
    if (-not (Get-Command $taskProgram -ErrorAction SilentlyContinue)) { throw "$taskProgram is unavailable." }
}
$taskRemoteDir = $RemoteExport.TrimEnd('/')
$taskExportName = ($taskRemoteDir -split '/')[-1]
if ($taskExportName -notmatch '^czsl_all_completed_[A-Za-z0-9_-]+$') { throw 'Unexpected export directory name.' }
$taskBase = [System.IO.Path]::GetFullPath($LocalRoot)
$taskLocal = Join-Path $taskBase $taskExportName
$taskIncoming = Join-Path $taskLocal '_incoming'
New-Item -ItemType Directory -Path $taskIncoming -Force | Out-Null
Write-Host "LOCAL_EXPORT=$taskLocal"

# Fetch the two small export descriptors in one scp session.
& scp -P $Port "${SshTarget}:${taskRemoteDir}/SHA256SUMS" "${SshTarget}:${taskRemoteDir}/EXPORT_COMPLETE.json" $taskIncoming
if ($LASTEXITCODE -ne 0) { throw 'Descriptor download failed. Rerun this command after the connection is restored.' }
$taskHashFile = Join-Path $taskIncoming 'SHA256SUMS'
$taskEntries = @()
foreach ($taskLine in Get-Content -LiteralPath $taskHashFile -Encoding ASCII) {
    if ($taskLine -notmatch '^([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9_.-]*)$') { throw "Invalid checksum entry: $taskLine" }
    $taskEntries += [pscustomobject]@{ Hash = $Matches[1].ToUpperInvariant(); Name = $Matches[2] }
}
if ($taskEntries.Count -eq 0) { throw 'Empty checksum manifest.' }
if (($taskEntries.Name | Select-Object -Unique).Count -ne $taskEntries.Count) { throw 'Duplicate checksum entries.' }
$taskDescriptorEntry = @($taskEntries | Where-Object Name -eq 'EXPORT_COMPLETE.json')
if ($taskDescriptorEntry.Count -ne 1) { throw 'Missing completion descriptor checksum.' }
$taskDescriptorPath = Join-Path $taskIncoming 'EXPORT_COMPLETE.json'
if ((Get-FileHash -LiteralPath $taskDescriptorPath -Algorithm SHA256).Hash -ne $taskDescriptorEntry[0].Hash) {
    throw 'Completion descriptor checksum mismatch.'
}
$taskInfo = Get-Content -LiteralPath $taskDescriptorPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $taskInfo.export_complete -or $taskInfo.export_name -ne $taskExportName) { throw 'Server export is incomplete or has a different identity.' }

$taskExpectedChecksumIdentity = (Get-FileHash -LiteralPath $taskHashFile -Algorithm SHA256).Hash
$taskAlreadyExtracted = $false
$taskExtractMarker = Join-Path $taskLocal 'LOCAL_EXTRACT_COMPLETE.json'
if (Test-Path -LiteralPath $taskExtractMarker -PathType Leaf) {
    $taskOld = Get-Content -LiteralPath $taskExtractMarker -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskAlreadyExtracted = $taskOld.checksums_file_hash -eq $taskExpectedChecksumIdentity -and (Test-Path -LiteralPath $taskOld.files_directory -PathType Container)
}
$taskArchiveSizes = @{}
foreach ($taskArchive in @($taskInfo.archive_files)) {
    if ($taskArchive.name -notmatch '^0[1-6]_[A-Za-z0-9_]+\.tar\.gz$') { throw 'Unexpected archive filename.' }
    if (@($taskEntries | Where-Object Name -eq $taskArchive.name).Count -ne 1) { throw 'Archive absent from checksum manifest.' }
    $taskArchiveSizes[$taskArchive.name] = [double]$taskArchive.bytes
}
$taskMissing = @()
$taskNeeded = [double]0
foreach ($taskEntry in $taskEntries) {
    $taskSaved = Join-Path $taskLocal $taskEntry.Name
    $taskStaged = Join-Path $taskIncoming $taskEntry.Name
    if ((Test-Path -LiteralPath $taskSaved -PathType Leaf) -and (Get-FileHash -LiteralPath $taskSaved -Algorithm SHA256).Hash -eq $taskEntry.Hash) {
        Write-Host "SKIP verified: $($taskEntry.Name)"
    } elseif ((Test-Path -LiteralPath $taskStaged -PathType Leaf) -and (Get-FileHash -LiteralPath $taskStaged -Algorithm SHA256).Hash -eq $taskEntry.Hash) {
        Write-Host "READY staged: $($taskEntry.Name)"
    } else {
        $taskMissing += "${SshTarget}:${taskRemoteDir}/$($taskEntry.Name)"
        if ($taskArchiveSizes.ContainsKey($taskEntry.Name)) { $taskNeeded += $taskArchiveSizes[$taskEntry.Name] }
    }
}
if (-not $DownloadOnly -and -not $taskAlreadyExtracted) {
    $taskNeeded += [double]$taskInfo.expanded_file_bytes + [double]$taskInfo.file_count * 4096
}
$taskVolume = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($taskLocal))
if ($taskVolume.IsReady -and $taskVolume.AvailableFreeSpace -lt $taskNeeded + 128MB) {
    throw 'Not enough local free space for remaining downloads and extraction. Use -LocalRoot on another drive, or -DownloadOnly.'
}
if ($taskMissing.Count -gt 0) {
    # All missing packages transfer in one session. An interruption is recoverable
    # by rerunning; valid completed files are skipped, the interrupted file restarts.
    & scp -P $Port @taskMissing $taskIncoming
    if ($LASTEXITCODE -ne 0) { throw 'Package download interrupted. Rerun the same command; valid completed packages will be skipped.' }
}
foreach ($taskEntry in $taskEntries) {
    $taskSaved = Join-Path $taskLocal $taskEntry.Name
    if ((Test-Path -LiteralPath $taskSaved -PathType Leaf) -and (Get-FileHash -LiteralPath $taskSaved -Algorithm SHA256).Hash -eq $taskEntry.Hash) { continue }
    $taskStaged = Join-Path $taskIncoming $taskEntry.Name
    if (-not (Test-Path -LiteralPath $taskStaged -PathType Leaf)) { throw "Missing file: $($taskEntry.Name)" }
    if ((Get-FileHash -LiteralPath $taskStaged -Algorithm SHA256).Hash -ne $taskEntry.Hash) { throw "Checksum mismatch: $($taskEntry.Name)" }
    Move-Item -LiteralPath $taskStaged -Destination $taskSaved -Force
    Write-Host "SHA256 OK: $($taskEntry.Name)"
}
Copy-Item -LiteralPath $taskHashFile -Destination (Join-Path $taskLocal 'SHA256SUMS') -Force
$taskChecksumIdentity = (Get-FileHash -LiteralPath (Join-Path $taskLocal 'SHA256SUMS') -Algorithm SHA256).Hash
$taskDownloadRecord = @{ download_complete = $true; server = $SshTarget; remote_export = $taskRemoteDir; checksums_file_hash = $taskChecksumIdentity; time = (Get-Date).ToString('o') }
$taskDownloadRecord | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskLocal 'LOCAL_DOWNLOAD_COMPLETE.json') -Encoding UTF8
Write-Host 'ALL_DOWNLOAD_HASHES_PASSED'
Write-Host "EXPECTED_STAGE_COMPLETE=$($taskInfo.expected_stage_complete)"
if ($DownloadOnly) { Write-Host "Saved archives at: $taskLocal"; return }

if (Test-Path -LiteralPath $taskExtractMarker -PathType Leaf) {
    $taskOld = Get-Content -LiteralPath $taskExtractMarker -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($taskOld.checksums_file_hash -eq $taskChecksumIdentity -and (Test-Path -LiteralPath $taskOld.files_directory -PathType Container)) {
        Write-Host "Extraction already complete: $($taskOld.files_directory)"
        return
    }
}
$taskFiles = Join-Path $taskLocal 'files'
if (Test-Path -LiteralPath $taskFiles) {
    $taskFiles = Join-Path $taskLocal ('files_retry_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + $PID)
}
New-Item -ItemType Directory -Path $taskFiles | Out-Null
foreach ($taskArchive in @($taskInfo.archive_files)) {
    if ($taskArchive.name -notmatch '^0[1-6]_[A-Za-z0-9_]+\.tar\.gz$') { throw 'Unexpected archive filename.' }
    $taskArchivePath = Join-Path $taskLocal $taskArchive.name
    $taskMembers = @(& tar -tzf $taskArchivePath)
    if ($LASTEXITCODE -ne 0) { throw "Archive listing failed: $($taskArchive.name)" }
    foreach ($taskMember in $taskMembers) {
        if ($taskMember -match '^[/\\]' -or $taskMember -match '^[A-Za-z]:' -or $taskMember -match '(^|[/\\])\.\.([/\\]|$)') {
            throw "Unsafe archive member: $taskMember"
        }
    }
    Write-Host "EXTRACTING $($taskArchive.name)"
    & tar -xzf $taskArchivePath -C $taskFiles
    if ($LASTEXITCODE -ne 0) { throw 'Extraction failed. Downloaded archives remain available; the partial extraction is retained.' }
}
@{ extract_complete = $true; checksums_file_hash = $taskChecksumIdentity; files_directory = $taskFiles; time = (Get-Date).ToString('o') } |
    ConvertTo-Json | Set-Content -LiteralPath $taskExtractMarker -Encoding UTF8
Write-Host 'LOCAL_BACKUP_COMPLETE'
Write-Host "FILES=$taskFiles"
