# VG-CPOD 方案 v1 — J1

冻结 OpenAI CLIP ViT-L/14，使用现有 UT-Zappos 缓存，联合训练静态与动态 CP 独立读出。

完整公式、实验设置、上传和下载命令见 [EXPERIMENT_J_v1.md](EXPERIMENT_J_v1.md)。

```bash
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/vgcpod_v1_roundJ
bash scripts/start_round_j_tmux.sh --gpus 1 2 3
tmux attach -t vgcpod_J
```

GPU1/2/3 分别运行 seed3351/3352/3353；共 3 次训练和 6 项 native/exact TRAIN 均值门控评估。

默认缓存：`/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14`。

默认输出：`/data/xhw/vgcpod_v1_runs/roundJ_20261021`。

I2/G0/G1 作为直接对照，D1 作为已有性能参照。J1 使用 τγ=0.56、η=0.25、ρ=0.15、β=0.075、rank=32，每 seed 10 epochs。新参数 `Wd` 负责动态项，原 `Wc` 负责静态项，共 147,456 个可训练参数。

整轮结束标志：`launcher.exitcode` 为 `0`，且日志出现 `ROUND_J_COMPLETE ... training_jobs 3 mean_gate_jobs 6`。

数学与数值测试：

```bash
python -m unittest discover -s tests -v
```
