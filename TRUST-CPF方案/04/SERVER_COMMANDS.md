# Compact 紧凑效用监督：仓库恢复与服务器运行

原始训练已完成，结果见 [results](results/README.md)。以下命令从仓库恢复同一执行版本；`CPF_CACHE` 指向已有、与指纹一致的 CLIP 特征缓存。原始个人部署命令另保存在执行包原稿中。

## 恢复与初始化

```bash
conda activate vgcpod_v1
cd "$HOME/czsl-research"
export CPF_REPO="$PWD"
git lfs pull --include="TRUST-CPF方案/**"
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage 04 --dest "$HOME/experiments/trust-04-replay"
export CPF_CACHE=/path/to/cache_ut_clip_l14
cd "$HOME/experiments/trust-04-replay/trustcpf_compact_v1"
python run.py setup --cache "$CPF_CACHE" \
  --source "$CPF_REPO/TRUST-CPF方案/02/results/raw_export/trustcpf_strong_v1" \
  --response-source "$CPF_REPO/TRUST-CPF方案/03/results/raw_export/trustcpf_modelcond_v1"
source scripts/env.sh
"$CPF_PY" -m unittest discover -s tests -v
```

初始化的 source/response-source 指向仓库内保存的前序结果，因此不要求原服务器的旧项目路径仍然存在。原始队列 GPU 分配为 0/1/2；需换卡时修改 `scripts/run_all.sh` 的 GPU 与 seed 对应。

## 训练、检查与打包

```bash
mkdir -p logs
nohup bash scripts/run_all.sh >> logs/queue.log 2>&1 < /dev/null &
tail -F logs/queue.log
```

队列结束后执行：

```bash
source scripts/env.sh
"$CPF_PY" run.py report
"$CPF_PY" scripts/check_complete.py
bash scripts/pack_results.sh
```

完成记录需达到：21/21 个阶段、9 份识别结果、6 份效用审计、18 份复用响应与6份策略结构。识别结果中 3 份 Drop 复用 02 阶段，新增 Compact / shuffled_utility 各三次适配。

## 保存已有归档到本地

Windows PowerShell 中，先将仓库克隆到当前工作目录，再执行：

```powershell
git lfs pull --include="TRUST-CPF方案/04/**"
$cpfDestination = Join-Path $PWD "../trust-04-saved-archive"
python tools/register_trustcpf_strong_series.py --stage 04 --restore-archive "$cpfDestination"
```

此命令还原本轮已经归档的原包；新训练结果按 `pack_results.sh` 输出的实际文件路径下载。
