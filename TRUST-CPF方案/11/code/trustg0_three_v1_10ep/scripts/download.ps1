param([string]$Server = 'xhw@211.87.232.122', [string]$LocalRoot = '', [int]$ExpectedPipelines = 9, [int]$ExpectedRows = 1620)
$ErrorActionPreference = 'Stop'
$g0Server = $Server
$g0Root = if ($LocalRoot) { $LocalRoot } else { Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'output' }
$g0Local = Join-Path $g0Root ('trustg0_three_results_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $g0Local -Force | Out-Null
$g0Archive = Join-Path $g0Local 'trustg0_three_v1_10ep_results.tar.gz'
$g0Remote = '/var/tmp/xhw/trustg0_three_v1_10ep_results.tar.gz'
scp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "${g0Server}:$g0Remote" "${g0Server}:$g0Remote.sha256" "$g0Local/"
if ($LASTEXITCODE -ne 0) { throw "Download incomplete. Files retained at $g0Local" }
$g0Expected = ((Get-Content -LiteralPath "$g0Archive.sha256" -Raw).Trim() -split '\s+')[0]
if ((Get-FileHash -LiteralPath $g0Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $g0Expected.ToLowerInvariant()) { throw 'SHA256 mismatch' }
tar -xzf $g0Archive -C $g0Local
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed' }
$g0Result = Join-Path $g0Local 'trustg0_three_v1_10ep/work/report.json'
$g0Report = Get-Content -LiteralPath $g0Result -Raw | ConvertFrom-Json
if (!$g0Report.complete -or $g0Report.pipelines.Count -ne $ExpectedPipelines -or $g0Report.metric_rows -ne $ExpectedRows) { throw 'Result counts differ from the expected experiment' }
Write-Host 'DOWNLOAD_SHA256_EXTRACTION_COMPLETE'
Write-Host "Results: $g0Local\trustg0_three_v1_10ep\work"
