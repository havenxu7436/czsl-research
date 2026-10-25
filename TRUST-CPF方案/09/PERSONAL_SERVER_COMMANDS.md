原始个人部署命令如下。从仓库恢复同一执行包的入口见 [SERVER_COMMANDS.md](SERVER_COMMANDS.md)。

# TRUST-CPF 第二批 10 epoch：完整实践命令

本批目录统一为 `trustcpf_stage_v1_10ep`，结果压缩包统一为 `trustcpf_stage_v1_10ep_results.tar.gz`。服务器是 `xhw@211.87.232.122`。

默认使用物理 GPU **0、1、2**，对应 seed **3351、3352、3353**。改用 1、2、3 时，只改启动命令末尾的三个数字。

## 1. 本机 Windows PowerShell：上传

代码 ZIP 已放在：

```text
C:\Users\llenovo\Documents\ChatGPT\USG\output\trustcpf_stage_v1_10ep.zip
```

在本机 PowerShell 执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\llenovo\Documents\ChatGPT\USG\trustcpf_stage_v1_10ep\scripts\upload.ps1"
ssh xhw@211.87.232.122
```

上传脚本会先核对本机 ZIP 的 SHA256，再上传 ZIP 与校验文件。也可以直接执行下面这段完整命令：

```powershell
$ErrorActionPreference = "Stop"
$cpfServer = "xhw@211.87.232.122"
$cpfZip = "C:\Users\llenovo\Documents\ChatGPT\USG\output\trustcpf_stage_v1_10ep.zip"
$cpfExpected = ((Get-Content -LiteralPath "$cpfZip.sha256" -Raw).Trim() -split '\s+')[0]
$cpfActual = (Get-FileHash -LiteralPath $cpfZip -Algorithm SHA256).Hash
if ($cpfActual -ne $cpfExpected) { throw "Local ZIP SHA256 mismatch" }
ssh $cpfServer "mkdir -p /data/xhw/experiments"
if ($LASTEXITCODE -ne 0) { throw "Remote directory creation failed" }
scp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $cpfZip "$cpfZip.sha256" "$($cpfServer):/data/xhw/experiments/"
if ($LASTEXITCODE -ne 0) { throw "Upload failed" }
ssh $cpfServer
```

SSH/SCP 输入密码时不显示字符，输入后按回车。

## 2. 服务器 Bash：解压和准备

连接服务器后执行这一整段：

```bash
set -e
cd /data/xhw/experiments
sha256sum -c trustcpf_stage_v1_10ep.zip.sha256
unzip -o trustcpf_stage_v1_10ep.zip
cd /data/xhw/experiments/trustcpf_stage_v1_10ep
conda activate vgcpod_v1
python -m pip install -r requirements.txt
python -c 'import sys, torch; print("PYTHON", sys.executable); print("TORCH", torch.__version__); print("CUDA", torch.cuda.is_available()); print("GPU_COUNT", torch.cuda.device_count())'
python run.py setup \
  --source /data/xhw/experiments/trustcpf_state_v1_10ep \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 \
  --work /data/xhw/experiments/trustcpf_stage_v1_10ep/work \
  --config /data/xhw/experiments/trustcpf_stage_v1_10ep/configs/default.json
mkdir -p logs
```

看到 `SETUP_COMPLETE` 后进入下一步。setup 使用服务器已完成的 **State 第一批**目录，包括 `work/utility/pooled_absolute/shared.pt`、三个 `runs/uniform/best.pt` 及 15 个教师；这里不使用旧 Competition 目录。

setup 会记录当前环境的 Python 绝对路径到 `server_env.sh`，后面的脚本自动读取。CLIP 特征直接复用现有缓存，不需要重提取或下载 CLIP 权重。

## 3. 服务器 Bash：三卡完整运行

使用 **0、1、2**：

```bash
cd /data/xhw/experiments/trustcpf_stage_v1_10ep
nohup bash scripts/run_all.sh 0 1 2 >> logs/queue.log 2>&1 &
echo $! > logs/queue.pid
tail -f logs/queue.log
```

若使用 **1、2、3**，上面的启动行替换为：

```bash
nohup bash scripts/run_all.sh 1 2 3 >> logs/queue.log 2>&1 &
```

选择一组启动即可。三卡先各自完成 5 个教师的 10 epoch 训练、双响应提取与 U-NLL 重测；全部完成后自动拟合一个共享预测器，再由三卡分别完成识别与 Q5 评分。

日志中的 `best_epoch` 可以小于 10，包括 0；它表示选出的权重。`TEACHER_EPOCH ... 10` 与 `contract.json` 的 `epochs_completed: 10` 表示训练预算已完成。

`Ctrl+C` 退出 `tail -f` 只停止看日志，后台队列继续运行。中断后重跑同一条 `nohup bash scripts/run_all.sh ...` 命令即可，已完成阶段复用，未完成教师从最近完整 epoch 恢复。

## 4. 进度与完成检查

以下命令在服务器项目目录执行：

```bash
cd /data/xhw/experiments/trustcpf_stage_v1_10ep
tail -n 30 logs/queue.log
tail -n 20 logs/seed3351.log
tail -n 20 logs/seed3352.log
tail -n 20 logs/seed3353.log
nvidia-smi
bash scripts/status.sh
```

完整结束会得到：

```text
STATUS 26 /26 teachers 15 /15 recognition 972 / 972 Q5 744 / 744 curves 165 /165 complete True
```

队列日志末尾会显示 `QUEUE_COMPLETE`。972 是 `3 seed ×27 策略 ×3 图 bank ×VAL/TEST ×CW/OW`；744 是 `3 seed ×31 策略 ×8 Q5 bank`；165 是 `15 教师 ×11 个记录点（epoch 0～10）`。

## 5. 服务器：汇总并打包

```bash
cd /data/xhw/experiments/trustcpf_stage_v1_10ep
bash scripts/pack_results.sh
ls -lh /data/xhw/experiments/trustcpf_stage_v1_10ep_results.tar.gz
```

打包时会输出 `PACK_PROGRESS`。等到 `PACK_COMPLETE` 和 `SHA256` 后开始下载。固定生成：

```text
/data/xhw/experiments/trustcpf_stage_v1_10ep_results.tar.gz
/data/xhw/experiments/trustcpf_stage_v1_10ep_results.tar.gz.sha256
```

包内包括完整结果、逐 epoch 曲线、日志、教师权重、测量统计与评分文件。源第一批目录保留在原处。

## 6. 本机 Windows PowerShell：下载、校验、解压

在本机另开 PowerShell，执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\llenovo\Documents\ChatGPT\USG\trustcpf_stage_v1_10ep\scripts\download.ps1"
```

脚本使用固定远程文件名；下载完成后核对 SHA256，再解压到新建的时间戳目录。结果位于：

```text
C:\Users\llenovo\Documents\ChatGPT\USG\output\stage_results\server_results_时间戳\trustcpf_stage_v1_10ep\work
```

也可使用下面这段完整命令，所有变量均在本段定义：

```powershell
$ErrorActionPreference = "Stop"
$cpfServer = "xhw@211.87.232.122"
$cpfLocal = "C:\Users\llenovo\Documents\ChatGPT\USG\output\stage_results"
$cpfArchive = Join-Path $cpfLocal "trustcpf_stage_v1_10ep_results.tar.gz"
$cpfHashFile = "$cpfArchive.sha256"
New-Item -ItemType Directory -Force -Path $cpfLocal | Out-Null
scp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$($cpfServer):/data/xhw/experiments/trustcpf_stage_v1_10ep_results.tar.gz" $cpfArchive
if ($LASTEXITCODE -ne 0) { throw "Archive download failed; rerun this block" }
scp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$($cpfServer):/data/xhw/experiments/trustcpf_stage_v1_10ep_results.tar.gz.sha256" $cpfHashFile
if ($LASTEXITCODE -ne 0) { throw "Checksum download failed" }
$cpfExpected = ((Get-Content -LiteralPath $cpfHashFile -Raw).Trim() -split '\s+')[0]
$cpfActual = (Get-FileHash -LiteralPath $cpfArchive -Algorithm SHA256).Hash
if ($cpfActual -ne $cpfExpected) { throw "SHA256 mismatch; download again" }
$cpfExtract = Join-Path $cpfLocal ("server_results_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $cpfExtract | Out-Null
tar -xzf $cpfArchive -C $cpfExtract
if ($LASTEXITCODE -ne 0) { throw "Extraction failed" }
Write-Host "DOWNLOAD_SHA256_EXTRACTION_COMPLETE"
Write-Host "Results: $cpfExtract\trustcpf_stage_v1_10ep\work"
```

## 7. 回传后优先分析的文件

| 文件 | 内容 |
|---|---|
| aggregates.json | 每 seed 先平均图 bank 后的三 seed 均值、SD 与逐 seed 数值 |
| summary.csv、paired.csv | 正式 VAL/TEST 的 CW/OW 指标与配对增量 |
| q5_decisions.csv、q5_paired.csv | Q5 实际决策评分与配对增量 |
| teacher_curves.csv、teachers.csv | 15 个教师的 10 epoch 曲线及选中轮次 |
| utility.csv、teacher_change.csv | Q5 效用预测、新旧监督变化 |
| response_repeats.csv、deployment.csv | 双响应稳定性、概率和结构统计 |
| status.json、logs/queue_result.json | 完整性和队列退出状态 |

主要比较是 `matched_single−old_single`（阶段匹配）、`matched_average−matched_single`（两次响应平均），同时报告它们相对 Uniform 与各自置换均值的改善。主表以真实识别指标为准，效用相关性和线性代理用于解释结果。

## 8. 已启动旧包：修复 CPU / CUDA 权重比较并续跑

如果日志在教师 epoch10 后出现 `Expected all tensors to be on the same device ... cpu and cuda:0 ... equal`，使用以下补丁。它修复冻结权重检查，并只迁移对应源码文件的版本摘要；已保存 checkpoint 保留。

本机 PowerShell：

```powershell
scp "C:\Users\llenovo\Documents\ChatGPT\USG\output\trustcpf_stage_device_fix.py" xhw@211.87.232.122:/data/xhw/experiments/trustcpf_stage_v1_10ep/scripts/fix_device_check.py
```

服务器执行：

```bash
cd /data/xhw/experiments/trustcpf_stage_v1_10ep
source scripts/env.sh
"$CPF_PY" scripts/fix_device_check.py --project "$PWD" --work "$CPF_WORK"
```

看到 `DEVICE_FIX_COMPLETE` 后，继续使用 1／2／3 卡：

```bash
nohup bash scripts/run_all.sh 1 2 3 >> logs/queue.log 2>&1 &
echo $! > logs/queue.pid
tail -f logs/seed3351.log
```

已完成 epoch10 的教师直接进入响应提取和效用测量，其余教师继续十轮训练。无需重新 setup 或删除 work。新版 ZIP 已包含此修复；已有旧 work 的运行使用此补丁完成版本迁移。

本机 GPU 回归验证通过：真实 CUDA 上新增教师训练 10 epoch、epoch3 断点恢复与连续训练逐项相同、epoch10 checkpoint 恢复不重复训练、版本迁移不改 checkpoint。
