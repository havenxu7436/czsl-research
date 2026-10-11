param([string]$DestinationRoot = 'D:\CodexData\CZSL')
$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent)).TrimEnd('\')
$destinationBase = [IO.Path]::GetFullPath($DestinationRoot).TrimEnd('\')
if (-not $destinationBase.StartsWith('D:\', [StringComparison]::OrdinalIgnoreCase) -or $destinationBase -eq 'D:') { throw 'Destination must be a named directory on D:.' }
if (-not (Test-Path -LiteralPath $destinationBase)) { New-Item -ItemType Directory -Path $destinationBase | Out-Null }
$freeBefore = (Get-PSDrive C).Free
$receipts = @()
foreach ($relative in @('output','tmp','archive_inputs','.git\lfs')) {
    $source = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $relative))
    $targetRelative = if ($relative -eq '.git\lfs') { 'git-lfs' } else { $relative }
    $target = [IO.Path]::GetFullPath((Join-Path $destinationBase $targetRelative))
    if (-not $source.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Source is outside workspace.' }
    if (-not $target.StartsWith($destinationBase + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Target is outside named destination.' }
    $sourceItem = Get-Item -LiteralPath $source -Force
    if ($sourceItem.LinkType -eq 'Junction' -and $sourceItem.Target -eq $target) { Write-Output "Already migrated: $relative"; continue }
    if ($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Preserve unexpected source link.' }
    if (Test-Path -LiteralPath $target) { throw "Preserve existing destination: $target" }
    $children = @(Get-ChildItem -LiteralPath $source -Recurse -Force)
    $nestedLinks = @($children | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })
    foreach ($link in $nestedLinks) {
        if ($link.LinkType -ne 'Junction' -or @($link.Target).Count -ne 1) { throw 'Preserve unsupported nested link.' }
        $linkTarget = [IO.Path]::GetFullPath([string]$link.Target)
        if (-not $linkTarget.StartsWith($source + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Preserve nested link outside this migration.' }
    }
    $sourceFiles = @($children | Where-Object { -not $_.PSIsContainer -and -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) })
    $sourceBytes = ($sourceFiles | Measure-Object Length -Sum).Sum
    Write-Output "Copying $relative to D: ($($sourceFiles.Count) files; $sourceBytes bytes)"
    & robocopy $source $target /E /COPY:DAT /DCOPY:T /R:1 /W:1 /XJ /NFL /NDL /NJH /NJS /NP
    if ($LASTEXITCODE -gt 7) { throw "Copy failed; C: originals retained: $relative" }
    $targetFiles = @(Get-ChildItem -LiteralPath $target -Recurse -Force -File)
    if ($targetFiles.Count -ne $sourceFiles.Count) { throw "File-count mismatch; C: originals retained: $relative" }
    # Preserve internal test-fixture junctions without copying their contents twice.
    # Their original C: targets remain valid through the top-level junction.
    foreach ($link in $nestedLinks) {
        $newLink = Join-Path $target $link.FullName.Substring($source.Length + 1)
        if (Test-Path -LiteralPath $newLink) { throw 'Unexpected copied link destination.' }
        New-Item -ItemType Junction -Path $newLink -Target ([string]$link.Target) | Out-Null
        $copiedLink = Get-Item -LiteralPath $newLink -Force
        if ($copiedLink.LinkType -ne 'Junction' -or $copiedLink.Target -ne $link.Target) { throw 'Nested junction verification failed.' }
    }
    Write-Output "Verifying SHA256 for $relative"
    foreach ($file in $sourceFiles) {
        $fileRelative = $file.FullName.Substring($source.Length + 1)
        $copied = Join-Path $target $fileRelative
        $copyInfo = Get-Item -LiteralPath $copied -Force
        if ($copyInfo.Length -ne $file.Length -or (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $copied -Algorithm SHA256).Hash) {
            throw "Hash mismatch; C: originals retained: $fileRelative"
        }
    }
    # Detect source changes during copying before replacing the original path.
    $sourceNow = @(Get-ChildItem -LiteralPath $source -Recurse -Force -File)
    if ($sourceNow.Count -ne $sourceFiles.Count) { throw 'Source changed during migration; originals retained.' }
    foreach ($file in $sourceFiles) {
        $now = Get-Item -LiteralPath $file.FullName -Force
        if ($now.Length -ne $file.Length -or $now.LastWriteTimeUtc -ne $file.LastWriteTimeUtc) { throw 'Source changed during migration; originals retained.' }
    }
    $backup = [IO.Path]::GetFullPath($source + '.migration-backup-20261011')
    if (-not $backup.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $backup)) { throw 'Unsafe or occupied migration backup.' }
    Move-Item -LiteralPath $source -Destination $backup
    try {
        New-Item -ItemType Junction -Path $source -Target $target | Out-Null
        $junction = Get-Item -LiteralPath $source -Force
        if ($junction.LinkType -ne 'Junction' -or $junction.Target -ne $target) { throw 'Junction verification failed.' }
    } catch {
        if (-not (Test-Path -LiteralPath $source)) { Move-Item -LiteralPath $backup -Destination $source }
        throw
    }
    # This exact backup was just hash-verified against D:. It is not a link.
    $backupItem = Get-Item -LiteralPath $backup -Force
    if ($backupItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Do not recursively remove a linked backup.' }
    foreach ($link in $nestedLinks) {
        $backupLink = [IO.Path]::GetFullPath((Join-Path $backup $link.FullName.Substring($source.Length + 1)))
        if (-not $backupLink.StartsWith($backup + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe nested backup link.' }
        $backupLinkItem = Get-Item -LiteralPath $backupLink -Force
        if ($backupLinkItem.LinkType -ne 'Junction' -or $backupLinkItem.Target -ne $link.Target) { throw 'Preserve changed backup link.' }
        [IO.Directory]::Delete($backupLink) # Remove only the link, never its target.
    }
    Remove-Item -LiteralPath $backup -Recurse -Force
    $receipts += [pscustomobject]@{source=$source;target=$target;files=$sourceFiles.Count;bytes=$sourceBytes;sha256_verified=$true;junction_verified=$true;nested_junctions_preserved=$nestedLinks.Count}
    Write-Output "Migrated and verified: $relative"
}
[pscustomobject]@{migrations=$receipts;free_before_bytes=$freeBefore;free_after_bytes=(Get-PSDrive C).Free;original_paths_preserved=$true;D_files_are_authoritative=$true} | ConvertTo-Json -Depth 5
