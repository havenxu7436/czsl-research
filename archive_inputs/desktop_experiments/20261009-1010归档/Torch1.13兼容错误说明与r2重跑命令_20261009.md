# Torch 1.13 兼容错误说明与 r2 重跑命令

日期：2026-10-09。

## 结论

这不是显存不足、Stage A 缓存损坏或模型训练退化。它是 pilot r1 检查点格式与服务器 Torch 1.13 安全读取器之间的兼容错误。

日志停在：

```text
cuda_resume_check
torch.load(..., weights_only=True)
WeightsUnpickler error: Unsupported operand 71
```

十进制 `71` 对应 pickle 的 `BINFLOAT` 指令。r1 的恢复自检把 optimizer/RNG 中的 Python 浮点数保存进内存检查点；本地 Torch 2.5 的 `weights_only` 读取器支持它，但服务器 Torch 1.13 的旧读取器不支持，所以安全读取主动拒绝。不能用 `weights_only=False` 绕过，因为那会扩大反序列化攻击面，也不能保证后续正式 `last.pt` 在旧环境中可靠恢复。

失败发生在正式流程之前：旧来源/Stage A 缓存校验已经通过；恢复自检只在内存中的临时 P_K head 上执行了一个诊断 optimizer step，随后读取失败，进程退出。Stage B 未开始，五组正式训练均为 0 个已提交 step，临时 head 没有保存到正式 checkpoint。日志返回 shell 提示符，说明该 Python 进程已经结束。

`PILOT_EXPORT_COMPLETE ... execution_passed False` 是预期的失败保护：程序保留诊断并导出取证 ZIP，没有冒充成功。请保留下列 r1 文件/目录，不删除、不覆盖，也不要用 r2 对它执行 `--resume`：

```text
/var/tmp/czsl_conditional_pilot_20261009
/var/tmp/czsl_conditional_pilot_20261009.zip
/var/tmp/czsl_conditional_pilot_20261009.zip.sha256
/tmp/czsl_conditional_pilot_20261009_driver.log
```

r2 将所有检查点中的 Python/NumPy 浮点元数据转换为带类型标记的 `float64` 标量张量，读取后再严格还原；仍坚持 `weights_only=True` 和 pickle protocol 2。相同编码覆盖恢复自检、`last.pt` 与 `best_auc.pt`，不是只修第一处报错。

本地已重新运行 24 项检查，全部通过；新增回归检查直接读取保存文件的 `data.pkl`，确认不再包含 `BINFLOAT/FLOAT` 指令。检查范围仍是 CPU 合成流程与格式兼容，服务器启动时还会在真实 Torch 1.13/CUDA 环境先执行恢复一致性自检；通过后才进入 Stage B。记录见 [local_test_report.json](C:/Users/llenovo/Documents/ChatGPT/CZSL/conditional_attribute_pilot_20261009/local_test_report.json)。

## 修复包

[conditional_attribute_pilot_release_20261009_r2.zip](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/conditional_attribute_pilot_release_20261009_r2.zip)

```text
SHA256 975015b8f58323e91faab2e9297f46bde271603573095db40ca4fb7e59f0bae7
大小 80852 bytes
```

r1 包及其哈希均保留；r2 是新文件，不覆盖 r1。

## 1. Windows PowerShell：上传 r2

```powershell
$pilotR2 = 'C:\Users\llenovo\Documents\ChatGPT\CZSL\output\conditional_attribute_pilot_release_20261009_r2.zip'
$pilotR2Sha = "$pilotR2.sha256"
if ((Get-FileHash -LiteralPath $pilotR2 -Algorithm SHA256).Hash.ToLowerInvariant() -ne '975015b8f58323e91faab2e9297f46bde271603573095db40ca4fb7e59f0bae7') {
    throw 'Pilot r2 ZIP mismatch.'
}
scp $pilotR2 $pilotR2Sha 'xhw@211.87.232.122:/tmp/'
if ($LASTEXITCODE -ne 0) { throw 'Upload failed.' }
ssh xhw@211.87.232.122
```

## 2. 服务器 Bash：用新目录重新启动

先确认 r1 已退出；以下命令应不再显示 r1 的 `pilot.py` 进程，仅命令自身的 `pgrep` 行可以忽略：

```bash
pgrep -af 'conditional_attribute_pilot_20261009/pilot.py'
nvidia-smi
```

下面示例使用物理 GPU 1；在首次运行整块命令前，可以按实际空闲卡修改 `CUDA_VISIBLE_DEVICES=1`。不要停止别人的任务。

```bash
(
  set -euo pipefail
  printf '%s  %s\n' '975015b8f58323e91faab2e9297f46bde271603573095db40ca4fb7e59f0bae7' '/tmp/conditional_attribute_pilot_release_20261009_r2.zip' | sha256sum -c -
  command -v tmux >/dev/null
  test -s /var/tmp/czsl_conditional_stageA_20261009/base/prepared.json
  test -s /var/tmp/czsl_conditional_stageA_20261009/base/train.pt
  test -s /var/tmp/czsl_conditional_stageA_20261009/base/val.pt
  for pilotR2Path in \
    /tmp/czsl_conditional_pilot_release_20261009_r2 \
    /tmp/czsl_conditional_pilot_20261009_r2_driver.log \
    /var/tmp/czsl_conditional_pilot_20261009_r2 \
    /var/tmp/czsl_conditional_pilot_20261009_r2.zip \
    /var/tmp/czsl_conditional_pilot_20261009_r2.zip.sha256 \
    /var/tmp/czsl_conditional_pilot_20261009_r2.zip.partial
  do
    if [ -e "$pilotR2Path" ]; then
      printf 'STOP: preserve existing r2 path: %s\n' "$pilotR2Path"
      exit 1
    fi
  done
  if tmux has-session -t cond_pilot_r2_1009 2>/dev/null; then
    printf 'STOP: r2 tmux session already exists.\n'
    exit 1
  fi
  /data/xhw/miniconda3/envs/cpf/bin/python -m zipfile -e \
    /tmp/conditional_attribute_pilot_release_20261009_r2.zip \
    /tmp/czsl_conditional_pilot_release_20261009_r2
  tmux new-session -d -s cond_pilot_r2_1009 \
    'CUDA_VISIBLE_DEVICES=1 /data/xhw/miniconda3/envs/cpf/bin/python -u /tmp/czsl_conditional_pilot_release_20261009_r2/conditional_attribute_pilot_20261009/pilot.py --work /var/tmp/czsl_conditional_pilot_20261009_r2 --export /var/tmp/czsl_conditional_pilot_20261009_r2.zip > /tmp/czsl_conditional_pilot_20261009_r2_driver.log 2>&1'
  printf 'STARTED: cond_pilot_r2_1009\n'
)
```

这里故意使用全新的解压目录、work、ZIP、日志和 tmux 名称；不是恢复 r1。原因是 r1 没有提交正式 epoch，而且 r2 的 release/protocol 哈希已经改变，混用会破坏实验身份。

## 3. 查看状态

```bash
tail -n 50 /tmp/czsl_conditional_pilot_20261009_r2_driver.log
pgrep -af 'conditional_attribute_pilot_20261009/pilot.py'
```

修复点通过后，不再出现 `Unsupported operand 71`，日志会继续出现：

```text
MECHANICS_BEGIN P_name ...
...
MECHANICS_COMPLETE: resetting every head before formal training
FORMAL_EPOCH_BEGIN P_name 1 / 10
```

首次看到 `FORMAL_EPOCH_BEGIN` 才表示五组正式 pilot 已经开始。全部完成应同时出现：

```text
PILOT_COMPLETE formal_steps 4500 BUDGET_GATE <True或False>
PILOT_EXPORT_COMPLETE ... execution_passed True
```

如果 r2 仍在真实 Torch 1.13 自检处失败，它同样会安全导出 `execution_passed False`；请保留并发送新日志，不要改成 `weights_only=False`。

## 4. 完成后下载 r2 结果

```powershell
$pilotR2Local = 'C:\Users\llenovo\Documents\ChatGPT\CZSL\output\conditional_attribute_pilot_results_20261009_r2'
New-Item -ItemType Directory -Path $pilotR2Local -Force | Out-Null
$pilotR2Result = Join-Path $pilotR2Local 'czsl_conditional_pilot_20261009_r2.zip'
if ((Test-Path -LiteralPath $pilotR2Result) -or (Test-Path -LiteralPath "$pilotR2Result.sha256")) {
    throw 'Preserve existing local r2 result.'
}
scp 'xhw@211.87.232.122:/var/tmp/czsl_conditional_pilot_20261009_r2.zip' 'xhw@211.87.232.122:/var/tmp/czsl_conditional_pilot_20261009_r2.zip.sha256' $pilotR2Local
if ($LASTEXITCODE -ne 0) { throw 'Download failed.' }
& 'C:\Users\llenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\llenovo\Documents\ChatGPT\CZSL\conditional_attribute_pilot_20261009\export_results.py' --verify-zip $pilotR2Result
if ($LASTEXITCODE -ne 0) { throw 'ZIP verification failed.' }
```

其余训练设计、五组定义、固定预算和性能门槛仍以 [Stage A 审阅与五组条件属性训练命令](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/conditional_attribute_stageA_results_20261009/StageA审阅与五组条件属性训练命令_20261009.md) 为准；本文件的 r2 上传/启动/下载命令替代其中 r1 对应命令。
