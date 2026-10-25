原始个人部署命令如下。从仓库恢复同一执行包的入口见 [SERVER_COMMANDS.md](SERVER_COMMANDS.md)。

# TRUST-CPF 第一批：上传、10 epoch 训练、结果下载

服务器：xhw@211.87.232.122。默认使用 **0／1／2 卡**，分别运行 seed 3351／3352／3353。

本版每个 seed 先从旧 reference 重新训练 10 epoch Uniform，随后冻结共同最佳权重完成所有策略比较。效用岭回归为闭式求解，不另设 epoch。

## 1. 在本机 PowerShell 上传

直接复制以下整段。压缩包与校验文件已在同一个本地目录。

```powershell
$cpfServer = "xhw@211.87.232.122"
$cpfZip = "C:\Users\llenovo\Documents\ChatGPT\USG\output\trustcpf_state_v1_10ep.zip"

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $cpfServer "mkdir -p /data/xhw/experiments"
if ($LASTEXITCODE -ne 0) { throw "服务器目录创建失败" }

scp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $cpfZip "$cpfZip.sha256" "$($cpfServer):/data/xhw/experiments/"
if ($LASTEXITCODE -ne 0) { throw "上传未完成，重新执行上传命令" }

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $cpfServer
```

输入密码时终端不显示字符。最后一条命令登录服务器。

## 2. 在服务器解压并准备

```bash
cd /data/xhw/experiments
sha256sum -c trustcpf_state_v1_10ep.zip.sha256
unzip -o trustcpf_state_v1_10ep.zip -d /data/xhw/experiments

cd /data/xhw/experiments/trustcpf_state_v1_10ep
conda activate vgcpod_v1
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -m pip install -r requirements.txt

python run.py setup \
  --source /data/xhw/experiments/trustcpf_r5_v1_10ep \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 \
  --work /data/xhw/experiments/trustcpf_state_v1_10ep/work
```

setup 导入已有 15 个教师、响应、U-NLL 标签和 reference，并统一完成四种岭回归拟合。正式 GPU 评分使用其中三个预测器。完成时显示 SETUP_COMPLETE，后续脚本会自动读取本次 Python 和路径。

代码包不重复打包原始图像与 CLIP 特征缓存；使用上一轮 R5 已运行成功的缓存。R5 的旧实验文件保持原样。

## 3. 在服务器用 0／1／2 卡运行完整队列

```bash
cd /data/xhw/experiments/trustcpf_state_v1_10ep
mkdir -p logs
nohup bash scripts/run_all.sh 0 1 2 > logs/queue.log 2>&1 &
tail -f logs/queue.log
```

三卡各一条完整队列；10 epoch 训练结束后，会自动继续提取响应、生成策略、执行 Q5 审计和冻结识别。

查看具体训练进度，另开一个服务器终端执行：

```bash
cd /data/xhw/experiments/trustcpf_state_v1_10ep
tail -f logs/seed3351.log
```

其余两条日志为 logs/seed3352.log、logs/seed3353.log。按 Ctrl+C 退出 tail，不会停止 nohup 任务。

若使用 1／2／3 卡，只把启动命令改成：

```bash
nohup bash scripts/run_all.sh 1 2 3 > logs/queue.log 2>&1 &
```

一次只启动一条队列。中断后在同一目录重新运行原启动命令，程序继续未完成 epoch 或阶段。

## 4. 检查三 seed 是否全部完成

```bash
cd /data/xhw/experiments/trustcpf_state_v1_10ep
bash scripts/status.sh
```

正式配置最终应显示：

```text
STATUS 17 /17 recognition 1512 / 1512 Q5 600 / 600 curves 33 /33 complete True
```

队列日志还会显示 QUEUE_COMPLETE。最佳 epoch 可能小于 10；实际训练到达第 10 epoch，由 training_contract.json 和 last.pt 记录。

查看卡占用：

```bash
nvidia-smi
```

## 5. 在服务器打包完整结果

确认 complete True 后运行：

```bash
cd /data/xhw/experiments/trustcpf_state_v1_10ep
bash scripts/pack_results.sh
```

打包会持续输出 PACK_PROGRESS，最后输出 PACK_COMPLETE。结果固定保存在：

```text
/data/xhw/experiments/trustcpf_state_v1_10ep_results.tar.gz
/data/xhw/experiments/trustcpf_state_v1_10ep_results.tar.gz.sha256
```

包内包含预测、效用模型、教师、Uniform 权重、标签、配置和日志。文件名采用英文，校验文件采用 LF 换行。

## 6. 回到本机 PowerShell 下载、校验、解压

以下整段独立可运行，不依赖前面会话中的变量：

```powershell
$ErrorActionPreference = "Stop"
$cpfServer = "xhw@211.87.232.122"
$cpfLocal = "C:\Users\llenovo\Documents\ChatGPT\USG\output\state_results"
New-Item -ItemType Directory -Force -Path $cpfLocal | Out-Null

$cpfArchive = Join-Path $cpfLocal "trustcpf_state_v1_10ep_results.tar.gz"
$cpfHashFile = "$cpfArchive.sha256"

scp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$($cpfServer):/data/xhw/experiments/trustcpf_state_v1_10ep_results.tar.gz" $cpfArchive
if ($LASTEXITCODE -ne 0) { throw "结果下载未完成，重新执行下载命令" }

scp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$($cpfServer):/data/xhw/experiments/trustcpf_state_v1_10ep_results.tar.gz.sha256" $cpfHashFile
if ($LASTEXITCODE -ne 0) { throw "校验文件下载失败" }

$cpfExpected = ((Get-Content -LiteralPath $cpfHashFile -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
$cpfActual = (Get-FileHash -LiteralPath $cpfArchive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($cpfActual -ne $cpfExpected) { throw "SHA256 不一致，请重新下载" }

$cpfExtract = Join-Path $cpfLocal ("server_results_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $cpfExtract | Out-Null
tar -xzf $cpfArchive -C $cpfExtract
if ($LASTEXITCODE -ne 0) { throw "解压失败" }

Write-Host "保存完成：$cpfExtract\trustcpf_state_v1_10ep\work"
```

每次解压进入新时间目录，保留之前的结果。原始压缩包也留在 state_results 中。

## 7. 保存后重点查看的文件

| 文件 | 内容 |
| --- | --- |
| work/status.json | 完整性和阶段数量 |
| work/summary.csv | 逐 seed、状态、策略、bank、split、world 的识别指标 |
| work/paired.csv | 相对 Uniform、五置换均值及独立／联合预测器的配对差值 |
| work/aggregates.json | 三 seed 汇总，以及先平均 bank 后的汇总 |
| work/q5_decisions.csv、q5_paired.csv | Q5 实际效用决策与配对比较 |
| work/utility.csv、utility/selection.json | 效用预测指标与共同 λ 选择 |
| work/training_curves.csv | 三 seed 的 epoch 0–10 曲线 |
| work/response_repeats.csv、response_transfer.csv | 重复响应与跨状态排序诊断 |
| work/deployment.csv、costs.csv | 类型概率、节点覆盖、原型位移与耗时 |
| work/seed3351/runs/uniform/best.pt、last.pt | seed3351 最佳／第 10 epoch 权重；另两个 seed 同结构 |

论文主比较使用 uniform_best 状态，同 seed 的所有方法共享完全相同的识别权重。reference 状态、错配响应和简单读出用于解释收益来源。

