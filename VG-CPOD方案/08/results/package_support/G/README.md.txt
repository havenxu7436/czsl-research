# VG-CPOD 方案 v1：G0／G1

两组固定 `rho=0.15`、`task_aux_beta=0.075`，使用冻结 OpenAI CLIP ViT-L/14 缓存。G0 设置 `gate_input=none`，G1 使用 `gate_input=local`、`gate_eta=0.25`。

在服务器原 conda 环境启动：

```bash
conda activate vgcpod_v1
bash scripts/start_round_g_tmux.sh --gpus 1 2 3
tmux attach -t vgcpod_G
```

GPU1/2/3 对应 seed3351/3352/3353，每张卡顺序训练 G0→G1，各 10 epochs，再执行 G1 的 native/exact 两项 TRAIN 均值门控评估。整轮共六项训练、六项门控评估，自动生成同 seed 比较和概率排序诊断。

完整公式与上传／训练／续训／下载命令见 [EXPERIMENT_G_v1.md](EXPERIMENT_G_v1.md)，排版预览见 [EXPERIMENT_G_v1.html](EXPERIMENT_G_v1.html)。

默认输出：`/data/xhw/vgcpod_v1_runs/roundG_20261020`。`launcher.exitcode` 为 0，且 `launcher.log` 出现 `ROUND_G_COMPLETE` 表示全部完成。
