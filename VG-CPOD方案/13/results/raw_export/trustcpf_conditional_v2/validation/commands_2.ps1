$cpfServer = "xhw@211.87.232.122"
$cpfLocal = "C:\Users\llenovo\Documents\ChatGPT\USG"
$cpfProject = Join-Path $cpfLocal "trustcpf_conditional_v2"
$cpfPython = "C:\Users\llenovo\anaconda3\envs\ai-lab\python.exe"

$cpfRemoteReply = ssh $cpfServer 'cat "$HOME/experiments/trustcpf_conditional_v2/latest_archive.txt"'
if ($LASTEXITCODE -ne 0 -or $null -eq $cpfRemoteReply) {
    throw "服务器结果路径未生成，请先完成第 5 节打包"
}
$cpfRemoteArchive = ($cpfRemoteReply -join "`n").Trim()
$cpfArchiveName = ($cpfRemoteArchive -split '/')[-1]
$cpfLocalArchive = Join-Path $cpfLocal $cpfArchiveName

scp "${cpfServer}:$cpfRemoteArchive" "$cpfLocalArchive"
if ($LASTEXITCODE -ne 0) { throw "下载未完成，请重新执行本节" }
scp "${cpfServer}:${cpfRemoteArchive}.sha256" "${cpfLocalArchive}.sha256"
if ($LASTEXITCODE -ne 0) { throw "校验文件下载未完成" }

$cpfExpectedHash = ((Get-Content -LiteralPath "${cpfLocalArchive}.sha256" -Raw) -split '\s+')[0]
$cpfActualHash = (Get-FileHash -LiteralPath $cpfLocalArchive -Algorithm SHA256).Hash
if ($cpfActualHash -ine $cpfExpectedHash) { throw "SHA256不一致，请重新下载压缩包" }

$cpfExtract = Join-Path $cpfLocal ($cpfArchiveName -replace '\.tar\.gz$', '')
& $cpfPython "$cpfProject\scripts\extract_results.py" "$cpfLocalArchive" "$cpfExtract"
if ($LASTEXITCODE -ne 0) { throw "解压未完成" }

Write-Host "结果已保存：$cpfExtract\trustcpf_conditional_v2\work"
Get-Content -LiteralPath "$cpfExtract\trustcpf_conditional_v2\work\status.json"