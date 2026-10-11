param([string]$Destination)
$ErrorActionPreference = 'Stop'
$workspaceRoot = $PSScriptRoot
$stagingRoot = Join-Path $workspaceRoot 'desktop_bundle_0601_20261011'
if (-not $Destination) {
    $Destination = Join-Path ([Environment]::GetFolderPath('Desktop')) '0601原设计复刻_代码与配置_20261011'
}
$Destination = [IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $Destination) { throw "Preserve existing folder: $Destination" }
$archiveSource = Join-Path $workspaceRoot 'output\0601_formula_release_20261011_r2.zip'
$expectedZip = '1516d14011c015754adfd54b421999ff3e7d05ee0a257f186e13e29a660c8c49'
if ((Get-FileHash -LiteralPath $archiveSource -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedZip) {
    throw 'Original r2 ZIP changed.'
}
$originalDesign = 'C:\Users\llenovo\Desktop\0601origin\ZSCL实验设计0601.md'
$expectedDesign = 'ebf696dec79c7788eb2ee02c095ca44828429a7e48225db77d56d9b08125ec0c'
if ((Get-FileHash -LiteralPath $originalDesign -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedDesign) {
    throw 'Original design version changed.'
}
New-Item -ItemType Directory -Path $Destination | Out-Null
foreach ($item in Get-ChildItem -LiteralPath $stagingRoot -Force) {
    Copy-Item -LiteralPath $item.FullName -Destination $Destination -Recurse
}
foreach ($relative in @('code','release','verification','environment','knowledge\text_inputs\ut-zap50k','knowledge\logic_reconstruction')) {
    $folderPath = Join-Path $Destination $relative
    if (-not (Test-Path -LiteralPath $folderPath)) { New-Item -ItemType Directory -Path $folderPath | Out-Null }
}
Copy-Item -LiteralPath $archiveSource -Destination (Join-Path $Destination 'release')
Copy-Item -LiteralPath "$archiveSource.sha256" -Destination (Join-Path $Destination 'release')
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::ExtractToDirectory($archiveSource, (Join-Path $Destination 'code'))
Copy-Item -LiteralPath $originalDesign -Destination (Join-Path $Destination 'docs\ZSCL实验设计0601.md')
Copy-Item -LiteralPath (Join-Path $workspaceRoot 'output\0601逐公式复刻审计_20261011.md') -Destination (Join-Path $Destination 'docs')
Copy-Item -LiteralPath (Join-Path $workspaceRoot 'output\0601_formula_delivery_verification_20261011.json') -Destination (Join-Path $Destination 'verification\r2_original_delivery_verification.json')
Copy-Item -LiteralPath (Join-Path $workspaceRoot 'output\0601_priority_results_20261011\extracted\pip_freeze.txt') -Destination (Join-Path $Destination 'environment\pip_freeze.actual_server.txt')
Copy-Item -LiteralPath (Join-Path $workspaceRoot 'output\conditional_attribute_pilot_results_20261009_r2\extracted\pilot\environment.json') -Destination (Join-Path $Destination 'environment\historical_actual_server.json')
$sourceText = Join-Path $workspaceRoot 'output\week5_evidence_20261005_verified\evidence\knowledge_source\ut-zap50k'
$targetText = Join-Path $Destination 'knowledge\text_inputs\ut-zap50k'
foreach ($name in @('attributes.txt','objects.txt','oi_dict.json','oe_dict.json','ae_dict.json','manifest.json','review_status.csv')) {
    Copy-Item -LiteralPath (Join-Path $sourceText $name) -Destination (Join-Path $targetText $name)
}
$targetLogic = Join-Path $Destination 'knowledge\logic_reconstruction'
Copy-Item -LiteralPath (Join-Path $workspaceRoot 'logiczsl_cpf_bf_20261008\assets\ut-zap50k\knowledge.json') -Destination (Join-Path $targetLogic 'knowledge.json')
Copy-Item -LiteralPath (Join-Path $workspaceRoot 'output\week5_evidence_20261005_verified\evidence\knowledge\ut-zap50k\labels.json') -Destination (Join-Path $targetLogic 'labels.json')
foreach ($name in @('build_bank.py','common.py')) {
    Copy-Item -LiteralPath (Join-Path $workspaceRoot "logiczsl_cpf_bf_20261008\$name") -Destination (Join-Path $targetLogic $name)
}
Copy-Item -LiteralPath (Join-Path $workspaceRoot 'logiczsl_cpf_bf_20261008\README.md') -Destination (Join-Path $targetLogic 'SOURCE_README_20261008.md')
$runbookSource = Join-Path $workspaceRoot 'output\0601主公式复刻_实践命令与时间预估_20261011.md'
$runbook = [IO.File]::ReadAllText($runbookSource)
$runbook = $runbook.Replace('C:/Users/llenovo/Documents/ChatGPT/CZSL/output/0601_formula_release_20261011_r2.zip.sha256', 'release/0601_formula_release_20261011_r2.zip.sha256')
$runbook = $runbook.Replace('C:/Users/llenovo/Documents/ChatGPT/CZSL/output/0601_formula_release_20261011_r2.zip', 'release/0601_formula_release_20261011_r2.zip')
$runbook = $runbook.Replace('C:\Users\llenovo\Documents\ChatGPT\CZSL\output\0601_formula_release_20261011_r2.zip', (Join-Path $Destination 'release\0601_formula_release_20261011_r2.zip'))
$utf8NoBom = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText((Join-Path $Destination 'RUNBOOK.md'), $runbook, $utf8NoBom)
$receipt = [ordered]@{
    created_at_local = (Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz')
    archive_scope = 'Selected UT-Zappos main formulas plus explicitly frozen CPF interfaces; not all original branches'
    desktop_directory = $Destination
    source_workspace = $workspaceRoot
    release_zip_sha256 = $expectedZip
    original_design_sha256 = $expectedDesign
    source_release_modified = $false
    server_connected = $false
    server_training_started_or_restarted = $false
    external_resources_excluded = @('dataset images','large GloVe file','ViT pretrained weights','original server cache tensors','training checkpoints')
}
[IO.File]::WriteAllText((Join-Path $Destination 'verification\archive_build_receipt.json'), ($receipt | ConvertTo-Json -Depth 10) + "`n", $utf8NoBom)
$checksums = [ordered]@{}
foreach ($file in Get-ChildItem -LiteralPath $Destination -Recurse -File | Sort-Object FullName) {
    $relative = $file.FullName.Substring($Destination.Length + 1).Replace('\','/')
    $checksums[$relative] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
$manifest = [ordered]@{ schema = '0601_desktop_bundle_v1'; files_sha256 = $checksums }
[IO.File]::WriteAllText((Join-Path $Destination 'bundle_manifest.json'), ($manifest | ConvertTo-Json -Depth 10) + "`n", $utf8NoBom)
Write-Output ([ordered]@{ destination=$Destination; manifest_files=$checksums.Count; release_sha256=$expectedZip } | ConvertTo-Json)
