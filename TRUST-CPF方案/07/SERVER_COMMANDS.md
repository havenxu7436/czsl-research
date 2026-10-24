# R5：仓库恢复、服务器训练与本地保存

[原个人上传、训练与下载命令](PERSONAL_SERVER_COMMANDS.md) 同时保留；以下从仓库恢复已交付版本。

## 服务器恢复与启动

```bash
conda activate vgcpod_v1
cd "$HOME/czsl-research"
export CPF_REPO="$PWD"
git lfs pull --include="TRUST-CPF方案/**"
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage 07 --dest "$HOME/experiments/trust-07-replay"
cd "$HOME/experiments/trust-07-replay/trustcpf_r5_v1_10ep"
python run.py setup \
  --source "$CPF_REPO/TRUST-CPF方案/06/results/raw_export/trustcpf_competition_v1_10ep" \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 \
  --work "$PWD/work"
source scripts/env.sh
CUDA_VISIBLE_DEVICES="" "$CPF_PY" -m unittest discover -s tests -v
mkdir -p logs
nohup bash scripts/run_all.sh 0 1 2 >> logs/queue.log 2>&1 < /dev/null &
tail -F logs/queue.log
```

已有教师、测量、响应及匹配识别结果按本轮协议自动复用。缓存需与 assets 指纹一致，三个 GPU 参数可按空闲卡调整。

## 检查与打包

```bash
source scripts/env.sh
"$CPF_PY" run.py report
"$CPF_PY" scripts/check_complete.py
bash scripts/pack_results.sh
```

下载新运行结果时，在 Windows 执行恢复包内 scripts/download_trustcpf_r5_v1_10ep.ps1；脚本读取服务器归档描述、核验 SHA256 并按 UTF-8 解压。

## 本地保存仓库内的已完成原始归档

在本地仓库根目录执行：

```powershell
git lfs pull --include="TRUST-CPF方案/07/**"
$cpfR5SavedArchive = Join-Path $PWD "../trust-07-saved-archive"
python tools/register_trustcpf_followup.py --stage 07 --restore-archive "$cpfR5SavedArchive"
```
