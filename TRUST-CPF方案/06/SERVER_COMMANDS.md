# Competition 三目标与 10 epoch：仓库恢复、服务器训练与本地保存

本页从仓库重建同一执行版本；[个人上传、训练和下载原命令](PERSONAL_SERVER_COMMANDS.md) 保留当时路径与操作说明。

## 服务器：从仓库恢复

```bash
conda activate vgcpod_v1
cd "$HOME/czsl-research"
export CPF_REPO="$PWD"
git lfs pull --include="TRUST-CPF方案/**"
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage 06 --dest "$HOME/experiments/trust-06-replay"
cd "$HOME/experiments/trust-06-replay/trustcpf_competition_v1_10ep"
python run.py setup \
  --source "$CPF_REPO/TRUST-CPF方案/05/results/raw_export/trustcpf_align_v1" \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 \
  --work "$PWD/work"
source scripts/env.sh
CUDA_VISIBLE_DEVICES="" "$CPF_PY" -m unittest discover -s tests -v
mkdir -p logs
nohup bash scripts/run_all.sh 0 2 3 >> logs/queue.log 2>&1 < /dev/null &
tail -F logs/queue.log
```

队列参数是三张物理卡与 seed3351/3352/3353 的对应，可按服务器空闲卡修改。缓存需与本阶段 assets 指纹一致。运行包含本批全部识别及 Q5 队列。

## 服务器：完成检查和打包

```bash
source scripts/env.sh
"$CPF_PY" run.py report
"$CPF_PY" scripts/check_complete.py
bash scripts/pack_results.sh
```

下载新运行结果时，在 Windows 执行恢复包内的 scripts/download_trustcpf_competition_v1_10ep.ps1；下载脚本根据服务器 latest_archive.json 取得归档名，校验 SHA256 后使用 UTF-8 解压。

## 本地：保存仓库内已完成的原始结果包

在本地仓库根目录执行：

```powershell
git lfs pull --include="TRUST-CPF方案/06/**"
$cpfSavedArchive = Join-Path $PWD "../trust-06-saved-archive"
python tools/register_trustcpf_followup.py --stage 06 --restore-archive "$cpfSavedArchive"
```
