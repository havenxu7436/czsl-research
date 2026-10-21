# VG-CPOD 方案 v1 · H1

冻结 OpenAI CLIP ViT-L/14，复用 UT-Zappos 缓存。H1 将 G1 门控输入改为减去 TRAIN 查询均值后的归一化视觉残差，η=0.25、ρ=0.15、β=0.075。

完整公式、三卡训练与结果下载命令见 [EXPERIMENT_H_v1.md](EXPERIMENT_H_v1.md)。

```bash
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/vgcpod_v1_roundH
bash scripts/start_round_h_tmux.sh --gpus 1 2 3
tmux attach -t vgcpod_H
```

三个 seed 各训练 10 epochs，随后自动完成六项 native／exact 均值门控评估和相对 G0／G1 的配对汇总。
