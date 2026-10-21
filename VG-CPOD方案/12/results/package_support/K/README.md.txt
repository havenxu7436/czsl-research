# VG-CPOD 方案 v1 · roundK

冻结 OpenAI CLIP ViT-L/14；UT-Zappos；K1/K2 调整独立动态读出 Wd 的学习率倍率为 0.3／0.1。

完整公式、配置、上传、tmux 训练、续训、汇总和下载命令见 [EXPERIMENT_K_v1.md](EXPERIMENT_K_v1.md)。带公式排版的阅读版为 [EXPERIMENT_K_v1.html](EXPERIMENT_K_v1.html)。

```bash
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/vgcpod_v1_roundK
bash scripts/start_round_k_tmux.sh --gpus 1 2 3
tmux attach -t vgcpod_K
```

GPU1／2／3 对应 seed3351／3352／3353，各卡 K1→K2，再完成两组 native/exact 均值评估。总计 6 次训练、60 个模型 epoch、12 项评估。

结果默认保存在 `/data/xhw/vgcpod_v1_runs/roundK_20261021`。完成标记为 `ROUND_K_COMPLETE ... training_jobs 6 mean_gate_jobs 12`，`launcher.exitcode` 为 `0`。
