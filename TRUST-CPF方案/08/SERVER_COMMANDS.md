# 从仓库恢复、训练与保存结果

原始 Windows 上传、服务器训练和下载的完整命令见 [个人部署命令](PERSONAL_SERVER_COMMANDS.md)。以下使用仓库归档作为来源；数据与 CLIP 缓存路径沿用已完成实验。

## 服务器恢复与训练

在已克隆的仓库根目录执行：

```bash
conda activate vgcpod_v1
export CPF_REPO="$PWD"
git lfs pull --include="TRUST-CPF方案/07/results/raw_export/**,TRUST-CPF方案/08/**"
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage 08 --dest "$HOME/experiments/trust-08-replay"
cd "$HOME/experiments/trust-08-replay/trustcpf_state_v1_10ep"
python -m pip install -r requirements.txt
python run.py setup \
  --source "$CPF_REPO/TRUST-CPF方案/07/results/raw_export/trustcpf_r5_v1_10ep" \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 \
  --work "$PWD/work"
mkdir -p logs
nohup bash scripts/run_all.sh 1 2 3 >> logs/queue.log 2>&1 < /dev/null &
tail -F logs/queue.log
```

原始命令也包含 GPU 0 / 1 / 2 的启动方式。完整结束后执行：

```bash
bash scripts/status.sh
bash scripts/pack_results.sh
```

## 取回已归档的本轮原始结果

在本地 PowerShell 的仓库根目录执行，选择一个新的还原目录：

```powershell
git lfs pull --include="TRUST-CPF方案/08/results/archives/**"
$cpfSavedArchive = Join-Path $PWD "../trust-08-saved-archive"
python tools/register_trustcpf_state_stage.py --stage 08 --restore-archive "$cpfSavedArchive"
```

恢复工具拼接分片并核对完整 SHA256，得到原始 TAR.GZ 与校验文件。

## 下载上述还原目录的新结果

以上目录的打包产物位于 `/data/xhw/experiments/trust-08-replay/`。原个人下载脚本对应原实验目录；使用上述还原目录时，在本地 PowerShell 运行：

```powershell
$cpfServer = "xhw@211.87.232.122"
$cpfRemote = "/data/xhw/experiments/trust-08-replay/trustcpf_state_v1_10ep_results.tar.gz"
$cpfLocal = Join-Path $PWD "../trust-08-rerun-results"
New-Item -ItemType Directory -Force -Path $cpfLocal | Out-Null
$cpfArchive = Join-Path $cpfLocal "trustcpf_state_v1_10ep_results.tar.gz"
scp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "${cpfServer}:$cpfRemote" "$cpfArchive"
if ($LASTEXITCODE -ne 0) { throw "Archive download failed" }
scp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "${cpfServer}:${cpfRemote}.sha256" "$cpfArchive.sha256"
if ($LASTEXITCODE -ne 0) { throw "Checksum download failed" }
$cpfExpected = ((Get-Content -LiteralPath "$cpfArchive.sha256" -Raw).Trim() -split '\s+')[0]
$cpfActual = (Get-FileHash -LiteralPath $cpfArchive -Algorithm SHA256).Hash
if ($cpfActual -ine $cpfExpected) { throw "Archive SHA256 mismatch" }
$cpfExtract = Join-Path $cpfLocal ("server_results_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $cpfExtract | Out-Null
tar -xzf "$cpfArchive" -C "$cpfExtract"
if ($LASTEXITCODE -ne 0) { throw "Archive extraction failed" }
Write-Host "Saved: $cpfExtract"
```
