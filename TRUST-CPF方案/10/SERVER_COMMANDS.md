# 从仓库恢复、服务器训练与保存

当时在本机上传、服务器训练与下载的具体命令保留在 [原始实践命令](PERSONAL_SERVER_COMMANDS.md)。下列命令从仓库重建执行包。

## 在服务器恢复并训练

在已经克隆的仓库根目录执行：

```bash
conda activate vgcpod_v1
export CPF_REPO="$PWD"
git lfs pull --include="TRUST-CPF方案/09/results/raw_export/**,TRUST-CPF方案/10/**"
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage 10 --dest "$HOME/experiments/trust-10-replay"
cd "$HOME/experiments/trust-10-replay/trustcpf_abc_v1_10ep"
python -m pip install -r requirements.txt
python run.py setup \
  --source "$CPF_REPO/TRUST-CPF方案/09/results/raw_export/trustcpf_stage_v1_10ep" \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 \
  --work "$PWD/work"
mkdir -p logs
nohup bash scripts/run_all.sh 1 2 3 > logs/queue.log 2>&1 < /dev/null &
echo $! > logs/queue.pid
tail -F logs/queue.log
```

源目录使用已完成 Stage（09）。需要 0/1/2 卡时，将 run_all.sh 的三个参数替换为 0 1 2。完成时状态为 36/36、recognition 936、Q4 3360、Q5 3504、directional 1056、curves 33。

```bash
bash scripts/status.sh
bash scripts/pack_results.sh
```

结果包位于该执行目录的上级。打包完成后下载结果包和 .sha256，并使用随包下载脚本核验、解压；完整 Windows PowerShell 命令见 [原始实践命令](PERSONAL_SERVER_COMMANDS.md)。

## 查看已归档结果

```bash
cd "$CPF_REPO"
python tools/register_trustcpf_abc.py --restore-results "$HOME/experiments/trust-10-saved"
```

训练使用数据和 CLIP 缓存作为外部输入；本仓库完整保存本次实验新增的权重、预测、测量及结果。
