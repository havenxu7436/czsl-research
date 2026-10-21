# VG-CPOD 方案 v1 · I1/I2

冻结 OpenAI CLIP ViT-L/14，复用 UT-Zappos 缓存和 H 组 TRAIN 均值。

- I1：TRAIN 残差保留原幅值，τγ=0.1。
- I2：TRAIN 残差 L2 归一化，τγ=0.56。
- 共同使用 η=0.25、ρ=0.15、β=0.075、rank32、10 epochs。

完整公式、上传、训练、续训和下载命令见 [EXPERIMENT_I_v1.md](EXPERIMENT_I_v1.md)。

```bash
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/vgcpod_v1_roundI
bash scripts/start_round_i_tmux.sh --gpus 1 2 3
tmux attach -t vgcpod_I
```

三卡各负责一个 seed 的 I1→I2 队列，完成六次训练、十二项 native／exact 均值门控评估和与 H1/G0/G1 的配对汇总。
