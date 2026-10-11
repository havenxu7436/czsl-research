param([string]$Bundle = 'C:\Users\llenovo\Desktop\0601原设计复刻_代码与配置_20261011')
$ErrorActionPreference = 'Stop'
$Bundle = [IO.Path]::GetFullPath($Bundle)
if (-not (Test-Path -LiteralPath (Join-Path $Bundle 'README.md'))) { throw 'Missing bundle.' }
$archivePath = "$Bundle.zip"
if ((Test-Path -LiteralPath $archivePath) -or (Test-Path -LiteralPath "$archivePath.sha256")) {
    throw 'Preserve existing desktop archive.'
}
$runtimePath = 'C:\Users\llenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = 'C:/Users/llenovo/Documents/ChatGPT/CZSL/tmp/0601_review_deps;C:/Users/llenovo/Documents/ChatGPT/CZSL/tmp/ekr_cpu_deps'
$env:PYTHONDONTWRITEBYTECODE = '1'
$utf8NoBom = [Text.UTF8Encoding]::new($false)
function Update-BundleManifest {
    $checksums = [ordered]@{}
    foreach ($file in Get-ChildItem -LiteralPath $Bundle -Recurse -File | Sort-Object FullName) {
        $relative = $file.FullName.Substring($Bundle.Length + 1).Replace('\','/')
        if ($relative -eq 'bundle_manifest.json') { continue }
        $checksums[$relative] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $manifest = [ordered]@{ schema='0601_desktop_bundle_v1'; files_sha256=$checksums }
    [IO.File]::WriteAllText((Join-Path $Bundle 'bundle_manifest.json'), ($manifest | ConvertTo-Json -Depth 10) + "`n", $utf8NoBom)
}
Update-BundleManifest
$verificationOutput = & $runtimePath (Join-Path $Bundle 'tools\verify_bundle.py') --bundle $Bundle --cpu-tests
if ($LASTEXITCODE -ne 0) { throw 'Bundle CPU/integrity verification failed; archive not created.' }
$verificationText = ($verificationOutput -join "`n") + "`n"
$verificationObject = $verificationText | ConvertFrom-Json
if (-not $verificationObject.passed) { throw 'Bundle reported failing checks.' }
[IO.File]::WriteAllText((Join-Path $Bundle 'verification\desktop_bundle_verification.json'), $verificationText, $utf8NoBom)
Update-BundleManifest
& $runtimePath (Join-Path $Bundle 'tools\verify_bundle.py') --bundle $Bundle
if ($LASTEXITCODE -ne 0) { throw 'Final bundle manifest failed.' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory($Bundle, $archivePath, [IO.Compression.CompressionLevel]::Optimal, $true)
$zipHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText("$archivePath.sha256", $zipHash + '  ' + [IO.Path]::GetFileName($archivePath) + "`n", $utf8NoBom)
$zip = [IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    $prefix = [IO.Path]::GetFileName($Bundle) + '/'
    $fileEntries = @($zip.Entries | Where-Object { -not $_.FullName.EndsWith('/') })
    $folderFiles = @(Get-ChildItem -LiteralPath $Bundle -Recurse -File)
    if ($fileEntries.Count -ne $folderFiles.Count) { throw 'Desktop ZIP file count differs.' }
    foreach ($entry in $fileEntries) {
        if (-not $entry.FullName.StartsWith($prefix, [StringComparison]::Ordinal)) { throw 'Unexpected desktop ZIP member.' }
        $relative = $entry.FullName.Substring($prefix.Length)
        $localFile = Join-Path $Bundle $relative
        $entryStream = $entry.Open()
        $hasher = [Security.Cryptography.SHA256]::Create()
        try { $entryHash = [BitConverter]::ToString($hasher.ComputeHash($entryStream)).Replace('-','').ToLowerInvariant() }
        finally { $entryStream.Dispose(); $hasher.Dispose() }
        if ($entryHash -ne (Get-FileHash -LiteralPath $localFile -Algorithm SHA256).Hash.ToLowerInvariant()) {
            throw "Desktop ZIP content mismatch: $relative"
        }
    }
} finally { $zip.Dispose() }
$totalBytes = ($folderFiles | Measure-Object -Property Length -Sum).Sum
Write-Output ([ordered]@{ bundle=$Bundle; archive=$archivePath; files=$folderFiles.Count; total_bytes=$totalBytes; zip_bytes=(Get-Item -LiteralPath $archivePath).Length; zip_sha256=$zipHash; desktop_zip_files_verified=$fileEntries.Count; server_training_started=$false } | ConvertTo-Json)
