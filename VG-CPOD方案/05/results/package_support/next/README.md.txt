# VG-CPOD v1：下一轮三卡实践命令

日期：2026-10-20。冻结 CLIP ViT-L/14，复用 UT-Zappos 缓存。启动包调用服务器原 C/D 训练代码和原参数。

| 物理 GPU | 任务 | 训练后处理 |
|---|---|---|
| GPU1 | D1 / seed3352 | 自动导出动态门控与 TRAIN 均值替换评估 |
| GPU2 | D1 / seed3353 | 自动导出动态门控与 TRAIN 均值替换评估 |
| GPU3 | C1 / seed3352 | 保存静态幅度对照结果 |

三项均为 10 epochs。D1/C1 的 rho=0.125；D1 的 eta=0.25。启动器自动跳过完成任务并续训中断任务。训练核心与配置保持原 v1 实现。

## 1. 本机 PowerShell 上传

将 `你的服务器IP` 替换为实际连接地址，以下使用 SSH 默认端口 22。

```powershell
$vgServer = "xhw@你的服务器IP"
scp "C:\Users\llenovo\Documents\ChatGPT\VG-CPOD\vgcpod_next_v1_20261020.zip" "${vgServer}:/data/xhw/vgcpod_project/"
```

## 2. 服务器启动 tmux 队列

```bash
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project
unzip -o vgcpod_next_v1_20261020.zip
cd /data/xhw/vgcpod_project/vgcpod_next_v1_20261020

bash start_tmux.sh --gpus 1 2 3
tmux attach -t vgcpod_next
```

按 Ctrl+b，松开，再按 d，离开 tmux 后队列继续。三个独立进程分别使用物理 GPU1/2/3，各自内部显示 cuda:0。两项 D1 训练各自在完成后于原卡执行均值替换评估。

调用的源目录：

- D1：`/data/xhw/vgcpod_project/vgcpod_v1_roundD`。
- C1：`/data/xhw/vgcpod_project/vgcpod_v1_roundC`。
- CLIP 缓存：`/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14`。

## 3. 查看进度和结束状态

主日志与三个训练日志：

```bash
tail -f /data/xhw/vgcpod_v1_runs/next_20261020/launcher.log
```

```bash
tail -n 10 /data/xhw/vgcpod_v1_runs/roundD_20261020/logs/D1_seed3352.log
tail -n 10 /data/xhw/vgcpod_v1_runs/roundD_20261020/logs/D1_seed3353.log
tail -n 10 /data/xhw/vgcpod_v1_runs/roundC_20261020/logs/C1_seed3352.log
cat /data/xhw/vgcpod_v1_runs/next_20261020/launcher.exitcode
```

`launcher.exitcode` 为 `RUNNING` 表示运行中，为 **0** 表示三项训练、两项 D1 均值替换与汇总完成。主日志同时显示 `NEXT_COMPLETE ... training_jobs 3 mean_gate_jobs 2`。`tail -f` 可按 Ctrl+C 退出；tmux 会话完成后保留终端。

本轮状态以 `next_20261020` 目录下的文件为准。`status.json` 记录三个任务及对应训练／评估状态；非零退出码的错误写在主日志和任务日志中。

## 4. 中断后继续

上一轮进程退出后，在服务器执行：

```bash
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/vgcpod_next_v1_20261020
VG_NEXT_SESSION=vgcpod_next_resume bash start_tmux.sh --gpus 1 2 3
tmux attach -t vgcpod_next_resume
```

已完成任务直接复用，已有 checkpoint 的未完成任务按原计划继续。

## 5. 结果位置

训练结果追加至原目录的新 seed：

- `roundD_20261020/D1/ut-zap50k/V1/seed3352`、`seed3353`。
- `roundC_20261020/C1/ut-zap50k/V1/seed3352`。
- 新 D1 均值替换：`roundD_20261020/mean_gate/D1/seed3352`、`seed3353`。

全部任务完成后自动更新：

- `roundD_20261020/round_d_summary.json`：C1/C2/D1 的完整结果。
- `roundD_20261020/round_d_paired_mean_sd.csv`：D1 三 seed 对 A5/C2 的配对均值及标准差。
- `roundD_20261020/mean_gate_effects.csv`：动态与 TRAIN 均值门控差异。
- `next_20261020/interaction_by_seed.csv`：seed3351、3352 的 `(D1-C2)-(C1-A5)`。
- `next_20261020/interaction_mean_sd.csv`：两 seed 交互差的均值与标准差。

## 6. 完成后下载到本机

先在服务器打包：

```bash
tar -czf /data/xhw/vgcpod_project/vgcpod_next_results_20261020.tar.gz \
  -C /data/xhw/vgcpod_v1_runs next_20261020 roundD_20261020 roundC_20261020 \
  -C /data/xhw/vgcpod_project vgcpod_next_v1_20261020 vgcpod_v1_roundD vgcpod_v1_roundC
```

再在本机 PowerShell 下载并解压：

```powershell
$vgServer = "xhw@你的服务器IP"
$vgLocal = "C:\Users\llenovo\Documents\ChatGPT\VG-CPOD"
scp "${vgServer}:/data/xhw/vgcpod_project/vgcpod_next_results_20261020.tar.gz" "$vgLocal\"
$vgExtract = Join-Path $vgLocal "server_results_next_20261020"
New-Item -ItemType Directory -Force -Path $vgExtract | Out-Null
tar -xzf "$vgLocal\vgcpod_next_results_20261020.tar.gz" -C "$vgExtract"
```

启动器经过 CPU 合成缓存的完整调度及重复启动检查。验证记录见 `validation.json` 和 `validation_cli.txt`。
