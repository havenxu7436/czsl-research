# VG-CPOD 方案 v1：F1

F1 在 E1 基础上将 `task_aux_beta` 从 0.1 调为 0.05，固定 `rho=0.15`，使用冻结 OpenAI CLIP ViT-L/14 缓存。

运行环境：原 `vgcpod_v1` conda 环境。上传解压后，在服务器包根目录执行：

```bash
conda activate vgcpod_v1
bash scripts/start_round_f_tmux.sh --gpus 1 2 3
tmux attach -t vgcpod_F
```

GPU1/2/3 对应 seed3351/3352/3353，每项 10 epochs；随后自动运行六项均值门控评估、生成同 seed 对比和加权损失汇总。

完整公式、上传／训练／续训／下载命令见 [EXPERIMENT_F_v1.md](EXPERIMENT_F_v1.md)，排版预览见 [EXPERIMENT_F_v1.html](EXPERIMENT_F_v1.html)。

默认输出：`/data/xhw/vgcpod_v1_runs/roundF_20261020`。`launcher.exitcode` 为 0 且 `launcher.log` 出现 `ROUND_F_COMPLETE` 表示完成。
