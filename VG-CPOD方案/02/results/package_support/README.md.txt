# VG-CPOD v1：下一轮 A 组实验

冻结 CLIP ViT-L/14，复用已有 UT-Zappos 特征缓存。实施计划、公式、上传和 tmux 三卡命令见 [EXPERIMENT_A_v1.md](EXPERIMENT_A_v1.md)。

```bash
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/vgcpod_v1_roundA
bash scripts/start_round_a_tmux.sh --gpus 1 2 3
tmux attach -t vgcpod_A
```

启动脚本将当前 conda 环境的绝对 Python 路径传入 tmux。GPU1 跑 A1→A2，GPU2 跑 A3→A4，GPU3 跑 A6→A5；每张卡一个进程。各组 seed3351、10 epochs。输出为 `/data/xhw/vgcpod_v1_runs/roundA_20261019`。

按 `Ctrl+b` 后松开，再按 `d` 退出 tmux，任务继续运行。结束时 `launcher.exitcode` 为 0，`round_a_status.json` 含六个完成状态。重启同一命令会跳过完成组并续训中断组。

```bash
python round_a.py summarize --out /data/xhw/vgcpod_v1_runs/roundA_20261019
```

`round_a_summary.csv` 汇总全部组，`round_a_summary.json` 附原始 V0/V1 同 seed 对照。训练权重仍按 VAL CW AUC_native 首次最大值选择；TEST 只用于最终导出。

环境沿用现有 `vgcpod_v1`，无须重新下载 CLIP 或重新编码。代码沿用原 PyTorch 1.13.1 / Python 3.8 兼容接口。本地验证和运行环境记录在 `validation/`。
