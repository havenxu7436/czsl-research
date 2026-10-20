# VG-CPOD v1：C 组静态幅度与中心化门控

完整公式、上传、训练和下载命令见 [EXPERIMENT_C_v1.md](EXPERIMENT_C_v1.md)。

```bash
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/vgcpod_v1_roundC
bash scripts/start_round_c_tmux.sh --gpus 1 2 3 --cases C1 C2 --seeds 3351 --b3-seed 3352
tmux attach -t vgcpod_C
```

GPU1 跑 C1/3351（静态 CP，ρ=0.125）；GPU2 跑 C2/3351（模式均值为 1 的局部弱门控）；GPU3 调用原 `/data/xhw/vgcpod_project/vgcpod_v1_roundB` 跑 B3/3352。三项均复用现有 CLIP ViT-L/14 缓存，各 10 epochs。

C1/C2 输出到 `/data/xhw/vgcpod_v1_runs/roundC_20261020`，B3 继续输出到原 B 组目录。联合汇总保存在 C 组目录，包含同 seed 的 A5/A6/B3/C1 对照。

按 Ctrl+b 后松开，再按 d，训练继续。总日志出现 `ROUND_C_COMPLETE ... jobs 3` 且 `launcher.exitcode` 为 0，表示队列成功结束。

```bash
python round_c.py summarize --out /data/xhw/vgcpod_v1_runs/roundC_20261020
```

后续只跑 C 组时使用 `--skip-b3`。当前 Python 3.8 / PyTorch 1.13 环境继续沿用；验证记录见 `validation/`。
