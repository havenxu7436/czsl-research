# VG-CPOD v1：B 组视觉读出与局部调制

冻结 CLIP ViT-L/14，复用现有 UT-Zappos 缓存。完整公式、代码改动、上传与下载命令见 [EXPERIMENT_B_v1.md](EXPERIMENT_B_v1.md)。

```bash
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/vgcpod_v1_roundB
bash scripts/start_round_b_tmux.sh --gpus 1 2 3 --cases B1 B2 B3 --seeds 3351
tmux attach -t vgcpod_B
```

GPU1=B1（混合读出），GPU2=B2（全局读出、局部门控 η=1），GPU3=B3（全局读出、局部门控 η=0.25）。每张卡一个进程，10 epochs，β=0.1，CP LR=3e-4。默认输出 `/data/xhw/vgcpod_v1_runs/roundB_20261019`。

按 Ctrl+b 后松开，再按 d，训练继续。完成时总日志打印 `ROUND_B_COMPLETE ... jobs 3`，`launcher.exitcode` 为 0。

```bash
python round_b.py summarize --out /data/xhw/vgcpod_v1_runs/roundB_20261019
```

汇总包含逐 seed 指标、配置均值与标准差、相对已有 A5/A6 的配对增量。三个配置内部均使用 V1 标识，由 `readout_feature`、`gate_input` 和 `gate_eta` 定义实际结构。

代码兼容原 Python 3.8 / PyTorch 1.13 环境。测试与本地 CPU 命令行验证记录位于 `validation/`。
