# TRUST-CPF 条件知识残差 v2：上传、训练、下载

服务器：`xhw@211.87.232.122`。正式实验为 Base / Knowledge / Shuffle × seed3351 / 3352 / 3353，共 9 次训练。分别使用物理 GPU **1 / 2 / 3**。每次 10 epochs。

代码包已包含三个 G0 最优权重、原始知识向量与固定打乱向量。复用历史全局 CLIP 缓存，原始图片与 CLIP 大权重无需重新上传。修订后的配方见 [修订方案](REVISED_PLAN.md)；该版本固定 CP 的属性／物体投影，训练条件读出与知识残差。

## 从仓库恢复部署包

本机 PowerShell 在克隆的仓库根目录执行；`$cpfRepo` 自动取当前目录：

```powershell
$cpfRepo = (Get-Location).Path
git lfs pull --include="TRUST-CPF方案/v2/**"
$cpfPython = "C:\Users\llenovo\anaconda3\envs\ai-lab\python.exe"
& $cpfPython tools/register_trustcpf_v2.py --check
$cpfReplay = Join-Path $env:USERPROFILE "experiments\trustcpf-v2-replay"
& $cpfPython tools/reproduce.py materialize --line TRUST-CPF方案 --stage v2 --dest "$cpfReplay"
```

`$cpfReplay` 使用新的绝对目录；运行包恢复为 `$cpfReplay\trustcpf_conditional_v2`。以下上传使用仓库中的原始部署 ZIP；第 6 节从该恢复目录读取解压工具。

## 1. 本机 PowerShell：上传

完整复制到本机 PowerShell：

```powershell
$cpfServer = "xhw@211.87.232.122"
$cpfZip = Join-Path $cpfRepo "TRUST-CPF方案\v2\results\deployment\trustcpf_conditional_v2.zip"

ssh $cpfServer 'mkdir -p "$HOME/experiments"'
scp "$cpfZip" "${cpfServer}:~/experiments/trustcpf_conditional_v2.zip"
if ($LASTEXITCODE -ne 0) { throw "上传未完成，请重新执行 scp" }

ssh $cpfServer
```

最后一行登录服务器。后面第 2–5 节在服务器终端执行。

## 2. 服务器：解压与环境配置

```bash
conda activate vgcpod_v1
python -m zipfile -e "$HOME/experiments/trustcpf_conditional_v2.zip" "$HOME/experiments"
cd "$HOME/experiments/trustcpf_conditional_v2"

python run.py setup
source scripts/env.sh
"$CPF_PY" -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())'
"$CPF_PY" -m unittest discover -s tests -v
```

`setup` 首先查找历史记录里的 `/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14`，也会检查之前 TRUST 项目的常用缓存位置。它核对本轮使用的 `g.npy`、文本和样本顺序，然后把当前 Python 的绝对路径、缓存路径及工作目录保存到 `server_env.sh`。后续脚本会自动读取，不再依赖旧终端里的 `CPF_PY`、`CACHE` 等变量。

若缓存实际被移动，指定它所在目录执行下面这一条，再继续第 3 节：

```bash
python run.py setup --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14
```

已有 `vgcpod_v1` 环境的 PyTorch 1.13.1、NumPy 1.24.4 满足依赖；这条训练路径只使用 PyTorch 和 NumPy。测试中使用合成数据的流水线分数不代表正式 CZSL 结果，结尾显示 `OK` 即测试通过。

## 3. 服务器：一次启动三张卡

```bash
cd "$HOME/experiments/trustcpf_conditional_v2"
mkdir -p logs
nohup bash scripts/run_all.sh >> logs/queue.log 2>&1 &
echo $! > logs/queue.pid
tail -f logs/queue.log
```

调度固定为：GPU1 跑 seed3351；GPU2 跑 seed3352；GPU3 跑 seed3353。每张卡内部依次执行 Base → Knowledge → Shuffle。内部 `cuda:0` 对应当前进程选中的那张物理卡。

查看详细训练日志，在另一个服务器终端执行：

```bash
cd "$HOME/experiments/trustcpf_conditional_v2"
tail -f logs/seed3351.log logs/seed3352.log logs/seed3353.log
```

`Ctrl+C` 结束的是 `tail` 查看，后台训练继续。查看 GPU：

```bash
nvidia-smi
```

每个 epoch 会打印本轮 `VAL_CW_AUC`、loss、alpha 和耗时。G0 起点的 VAL CW AUC 写在各目录 `initial.json` 中，历史三个 seed 分别约为 43.907、41.632、44.491。

## 4. 服务器：检查完成、恢复与汇总

```bash
cd "$HOME/experiments/trustcpf_conditional_v2"
source scripts/env.sh
"$CPF_PY" run.py status --work "$CPF_WORK"
tail -n 12 logs/queue.log
```

全部完成时显示 `COMPLETED 9 / 9 MISSING []`，总日志末尾为 `QUEUE_COMPLETE`。每组最终目录中都有 `DONE.json`；它在最优权重、完整 VAL/TEST 评价和逐图像预测都保存后才生成。

队列因断连以外的原因中断后，重新执行第 3 节的启动命令即可。已完成的模型直接跳过，未完成模型从最近完整 epoch 续训。若同一队列还在运行，锁会阻止重复启动。

汇总结果：

```bash
cd "$HOME/experiments/trustcpf_conditional_v2"
source scripts/env.sh
"$CPF_PY" run.py report --work "$CPF_WORK"
cat "$CPF_WORK/aggregates.json"
cat "$CPF_WORK/paired_comparisons.json"
```

`summary.csv` 保存逐 seed、逐组、逐 split/world 指标；`aggregates.json` 给出三 seed 的 AUC 均值和样本标准差；`paired_comparisons.json` 给出 Knowledge−Base、Knowledge−Shuffle 的逐 seed AUC 差值。各模型 `metrics.json` 同时记录固定 VAL bias 的 HM、未见条件属性准确率与 margin、物体分支准确率、联合物体命中率、alpha、耗时和显存。

## 5. 服务器：打包完整结果

九组完成后执行：

```bash
cd "$HOME/experiments/trustcpf_conditional_v2"
bash scripts/pack_results.sh
cat latest_archive.txt
```

打包包括代码、配置、初始权重、九组训练权重、日志、汇总、完整预测和样本标识。缓存不重复打包。压缩包位于 `$HOME/experiments/`，名字包含打包时间；路径写入 `latest_archive.txt`，另保存同名 `.sha256` 校验文件。归档内部文件名统一使用英文。

## 6. 本机 PowerShell：下载、校验、解压

新开本机 PowerShell，复制整个代码块：

```powershell
$cpfServer = "xhw@211.87.232.122"
$cpfLocal = "C:\Users\llenovo\Documents\ChatGPT\USG"
$cpfProject = Join-Path $env:USERPROFILE "experiments\trustcpf-v2-replay\trustcpf_conditional_v2"
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
```

下载脚本使用本文首节恢复出的 `trustcpf-v2-replay/trustcpf_conditional_v2/scripts/extract_results.py`。采用 UTF-8 解压与 SHA256 校验，处理前次下载中断、Windows tar 中文路径报错这两类问题。
