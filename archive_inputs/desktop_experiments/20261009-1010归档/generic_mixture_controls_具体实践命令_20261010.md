# Generic 温和概率混合对照：具体实践命令

日期：2026-10-10。

本轮新增训练只有 `G_name`、`G_perm1`、`G_perm2` 三组。三组都使用原 `P_generic` 拼接 MLP、相同初始化、相同训练顺序和十轮预算，只改变新增语义槽内容。已有正确知识 `G_K` 在服务器通过初始化、第一轮训练轨迹/分数和最后 checkpoint 实际 forward 精确复核后复用，不新增正式训练。

正式新增预算：3 组 × 10 epoch × 90 step = 2700 optimizer step。读出统一固定为 `0.75*q_base + 0.25*q_head`；alpha 不搜索。训练损失仍是原未混合 head 的条件 CE。视觉骨干更新、图片解码和 TEST 评估均为 0。

发布包：

```text
C:\Users\llenovo\Documents\ChatGPT\CZSL\output\generic_mixture_controls_release_20261010.zip
SHA256 292643510de430d63427b2a1cab18d2290a583a73ff3bb59c997ff97887e0736
大小 105634 bytes
```

## 1. Windows PowerShell：核验并上传

在本地 PowerShell 中一次性执行：

```powershell
$release = 'C:\Users\llenovo\Documents\ChatGPT\CZSL\output\generic_mixture_controls_release_20261010.zip'
$expected = '292643510de430d63427b2a1cab18d2290a583a73ff3bb59c997ff97887e0736'
$actual = (Get-FileHash -LiteralPath $release -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "Release SHA256 mismatch: $actual" }

ssh 'xhw@211.87.232.122' 'if [ -e /tmp/generic_mixture_controls_release_20261010.zip ]; then echo "STOP: remote release already exists"; exit 1; fi'
if ($LASTEXITCODE -ne 0) { throw 'Remote preflight failed; nothing uploaded.' }

scp $release 'xhw@211.87.232.122:/tmp/generic_mixture_controls_release_20261010.zip'
if ($LASTEXITCODE -ne 0) { throw 'Upload failed.' }
```

若远端同名发布包已存在，命令会停止而不是覆盖。先不要删除或改名复用旧目录；把输出发给我核对。

## 2. 服务器：只启动一次

SSH 登录服务器，在 bash 中完整粘贴以下代码块：

```bash
(
  set -euo pipefail

  printf '%s  %s\n' \
    '292643510de430d63427b2a1cab18d2290a583a73ff3bb59c997ff97887e0736' \
    '/tmp/generic_mixture_controls_release_20261010.zip' | sha256sum -c -

  command -v tmux >/dev/null
  test -x /data/xhw/miniconda3/envs/cpf/bin/python
  test -s /var/tmp/czsl_conditional_stageA_20261009/base/train.pt
  test -s /var/tmp/czsl_conditional_stageA_20261009/base/val.pt
  test -s /var/tmp/czsl_conditional_pilot_20261009_r2/protocol.json
  test -s /var/tmp/czsl_conditional_pilot_20261009_r2/runs/P_generic/last.pt
  test -s /var/tmp/czsl_conditional_pilot_20261009_r2/runs/P_generic/history.json

  for targetPath in \
    /tmp/czsl_generic_mixture_controls_release_20261010 \
    /tmp/czsl_generic_mixture_controls_20261010_driver.log \
    /var/tmp/czsl_generic_mixture_controls_20261010 \
    /var/tmp/czsl_generic_mixture_controls_20261010.zip \
    /var/tmp/czsl_generic_mixture_controls_20261010.zip.sha256 \
    /var/tmp/czsl_generic_mixture_controls_20261010.zip.partial
  do
    if [ -e "$targetPath" ]; then
      printf 'STOP: preserve existing path: %s\n' "$targetPath"
      exit 1
    fi
  done

  if tmux has-session -t generic_mix_1010 2>/dev/null; then
    printf 'STOP: tmux session generic_mix_1010 already exists.\n'
    exit 1
  fi

  /data/xhw/miniconda3/envs/cpf/bin/python -m zipfile -e \
    /tmp/generic_mixture_controls_release_20261010.zip \
    /tmp/czsl_generic_mixture_controls_release_20261010

  tmux new-session -d -s generic_mix_1010 \
    'CUDA_VISIBLE_DEVICES=1 /data/xhw/miniconda3/envs/cpf/bin/python -u /tmp/czsl_generic_mixture_controls_release_20261010/generic_mixture_controls_20261009/run_controls.py --work /var/tmp/czsl_generic_mixture_controls_20261010 --export /var/tmp/czsl_generic_mixture_controls_20261010.zip > /tmp/czsl_generic_mixture_controls_20261010_driver.log 2>&1'

  printf 'STARTED: generic_mix_1010\n'
)
```

必须使用 `CUDA_VISIBLE_DEVICES=1`：已有正确知识 generic 是在这个环境身份下训练的，新程序会核对原 Python/Torch/NumPy/SciPy、CUDA 可见设备和 GPU 名称。若环境不同，程序会拒绝将旧正确知识组当成严格配对参考。

重复粘贴会因已有目录或 tmux session 输出 `STOP`，不代表第一次启动失败。

## 3. 查看状态

```bash
tail -n 60 /tmp/czsl_generic_mixture_controls_20261010_driver.log
pgrep -af 'generic_mixture_controls_20261009/run_controls.py'
tmux has-session -t generic_mix_1010 && printf 'TMUX_RUNNING\n' || printf 'TMUX_ENDED\n'
```

正常阶段依次包括：

```text
VERIFYING_FIXED_INPUTS_AND_OLD_GENERIC
REFERENCE_FIRST_EPOCH_REPRODUCE_BEGIN
REFERENCE_REUSE_VERIFIED
REFERENCE_READOUT_EPOCH 0 ... 10
GENERIC_MECHANICS_BEGIN / COMPLETE
FORMAL_EPOCH_BEGIN/END G_name
FORMAL_EPOCH_BEGIN/END G_perm1
FORMAL_EPOCH_BEGIN/END G_perm2
```

其中 `REFERENCE_FIRST_EPOCH_REPRODUCE_BEGIN` 会额外执行旧正确知识 generic 的第一轮作为身份复核，但该诊断权重会丢弃，不计入正式 2700 步，也不会继续用于训练。

完整成功必须同时看到：

```text
GENERIC_CONTROLS_COMPLETE new_formal_steps 2700 GENERIC_GATE <True或False>
GENERIC_CONTROLS_EXPORT_COMPLETE ... execution_passed True
```

`GENERIC_GATE False` 仍是有效完成结果，只表示正确知识没有超过同结构名称/错配对照。不要据此改 alpha、删对照或只重跑某一组。

## 4. 完成后核验服务器结果

```bash
set -euo pipefail
test -s /var/tmp/czsl_generic_mixture_controls_20261010.zip
test -s /var/tmp/czsl_generic_mixture_controls_20261010.zip.sha256
cd /var/tmp
sha256sum -c czsl_generic_mixture_controls_20261010.zip.sha256
/data/xhw/miniconda3/envs/cpf/bin/python \
  /tmp/czsl_generic_mixture_controls_release_20261010/generic_mixture_controls_20261009/export_controls.py \
  --verify-zip /var/tmp/czsl_generic_mixture_controls_20261010.zip
```

最后应出现 `GENERIC_CONTROLS_ZIP_VERIFIED ... members`。

## 5. Windows PowerShell：下载与本地核验

服务器完整结束后，在本地 PowerShell 执行：

```powershell
$localDir = 'C:\Users\llenovo\Documents\ChatGPT\CZSL\output\generic_mixture_controls_results_20261010'
New-Item -ItemType Directory -Path $localDir -Force | Out-Null
$localZip = Join-Path $localDir 'czsl_generic_mixture_controls_20261010.zip'
$localSha = "$localZip.sha256"
if ((Test-Path -LiteralPath $localZip) -or (Test-Path -LiteralPath $localSha)) {
    throw 'Preserve existing local result download.'
}

scp `
  'xhw@211.87.232.122:/var/tmp/czsl_generic_mixture_controls_20261010.zip' `
  'xhw@211.87.232.122:/var/tmp/czsl_generic_mixture_controls_20261010.zip.sha256' `
  $localDir
if ($LASTEXITCODE -ne 0) { throw 'Download failed.' }

& 'C:\Users\llenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  'C:\Users\llenovo\Documents\ChatGPT\CZSL\generic_mixture_controls_20261009\export_controls.py' `
  --verify-zip $localZip
if ($LASTEXITCODE -ne 0) { throw 'Local ZIP verification failed.' }
```

下载完成后告诉我“generic controls 已下载”，我会独立核对 4 组全部 44 份读出、2700 正式步、原正确知识轨迹复用证据、选中 epoch、CW/OW 保护指标，以及正确知识相对名称和两种错配的差值。

## 6. 仅在真实中断后恢复

只有原进程已退出、日志没有成功导出，并且工作目录已经存在时，才使用下列命令。不要在原进程仍运行时启动恢复；排他锁也会阻止双写。

```bash
(
  set -euo pipefail
  test -s /var/tmp/czsl_generic_mixture_controls_20261010/protocol.json
  if pgrep -af 'generic_mixture_controls_20261009/run_controls.py' | grep -v grep; then
    printf 'STOP: original process is still running.\n'
    exit 1
  fi

  for resumePath in \
    /tmp/czsl_generic_mixture_controls_20261010_resume1.log \
    /var/tmp/czsl_generic_mixture_controls_20261010_resume1.zip \
    /var/tmp/czsl_generic_mixture_controls_20261010_resume1.zip.sha256 \
    /var/tmp/czsl_generic_mixture_controls_20261010_resume1.zip.partial
  do
    if [ -e "$resumePath" ]; then
      printf 'STOP: preserve existing resume path: %s\n' "$resumePath"
      exit 1
    fi
  done

  if tmux has-session -t generic_mix_resume1_1010 2>/dev/null; then
    printf 'STOP: resume tmux session already exists.\n'
    exit 1
  fi

  tmux new-session -d -s generic_mix_resume1_1010 \
    'CUDA_VISIBLE_DEVICES=1 /data/xhw/miniconda3/envs/cpf/bin/python -u /tmp/czsl_generic_mixture_controls_release_20261010/generic_mixture_controls_20261009/run_controls.py --resume --work /var/tmp/czsl_generic_mixture_controls_20261010 --export /var/tmp/czsl_generic_mixture_controls_20261010_resume1.zip > /tmp/czsl_generic_mixture_controls_20261010_resume1.log 2>&1'

  printf 'RESUME_STARTED: generic_mix_resume1_1010\n'
)
```

恢复从各组最后一个完整原子提交的 `last.pt` 接续 model、optimizer 和 RNG；中断轮从上一完整 epoch 重算。每次明确恢复都会重复三步丢弃式 checkpoint 自检，因此实际调用数可能多于正式 2700 步，汇总会分别计数。

若恢复成功，下载服务器上的 `czsl_generic_mixture_controls_20261010_resume1.zip` 及 sidecar，并在本地也使用新文件名，不能覆盖首次导出的失败取证包。

## 7. 本包已完成的本地检查

- 28 项 CPU 合成测试全部通过。
- 检查四种内容模式共用原 generic 的 60864 个有效参数和完全相同的初始参数状态。
- 检查 alpha=0.25 条件概率混合、完整联合分布恒等式、对象边缘保持和总变差上界。
- 实际执行两轮三种对照之一、checkpoint 回放和显式断点恢复。
- 使用原生 evaluator，SHA256 为 `24b48c4d48e3475baec0a656fca194cdbc6d4b4e32e87c0f22cda3daf586d978`。
- 发布 ZIP 21 个成员，CRC、成员哈希、Python 3.8 语法和清单均通过。

这些是本地 CPU 与历史文件身份检查，不冒充服务器 CUDA 正式训练结果。服务器启动后的 `REFERENCE_REUSE_VERIFIED` 和最终 `execution_passed True` 才确认实际环境中的配对执行完成。
