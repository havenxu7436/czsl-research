# VG-CPOD 方案 v1：E 组

冻结 CLIP ViT-L/14，复用现有 UT-Zappos 缓存。E1 沿用 D1，仅将 rho 改为 0.15。

默认 GPU1/2/3 分别训练 seed3351/3352/3353，各 10 epochs。训练器保存 native／exact 独立最优权重及对应评价，随后完成六项 TRAIN 均值门控评估，并汇总 C2/D1/E1 三档 rho。

```bash
bash scripts/start_round_e_tmux.sh --gpus 1 2 3
tmux attach -t vgcpod_E
```

完整公式、上传、启动、续训、结果与下载命令见 [EXPERIMENT_E_v1.md](EXPERIMENT_E_v1.md) 或 [HTML 版](EXPERIMENT_E_v1.html)。

默认输出：`/data/xhw/vgcpod_v1_runs/roundE_20261020`。
