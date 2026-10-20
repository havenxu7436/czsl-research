# VG-CPOD 方案 v1：D 组

冻结 CLIP ViT-L/14，复用已有 UT-Zappos 特征缓存。

默认 GPU1 补 C2/3352、GPU2 补 C2/3353、GPU3 训练 D1/3351。D1 是 C2 仅将 rho 调整为 0.125。随后导出 C2 三 seed 和 D1 同 checkpoint 的动态门控与 TRAIN 均值门控评估。

在服务器已有 `vgcpod_v1` 环境中运行：

```bash
bash scripts/start_round_d_tmux.sh --gpus 1 2 3
tmux attach -t vgcpod_D
```

完整公式、上传、启动、进度、续训和下载步骤见 [EXPERIMENT_D_v1.md](EXPERIMENT_D_v1.md)；可直接浏览 [EXPERIMENT_D_v1.html](EXPERIMENT_D_v1.html)。

默认沿用 `/data/xhw/vgcpod_project/vgcpod_v1_roundC` 原代码及 C 组输出，D 组结果位于 `/data/xhw/vgcpod_v1_runs/roundD_20261020`。
