# VG-CPOD方案v1

基于冻结的 **CLIP ViT-L/14 图像与文本双编码器**，实现修订后的视觉引导类别低秩算子、类别伙伴聚合和全部 9 组 UT-Zappos 对照实验。

从 [服务器实践与实现说明](VG-CPOD方案v1.md) 开始；完整推导见 [数学与实验设计](数学与实验设计_v1.md)。

| 内容 | 入口 |
|---|---|
| 数据和权重核对、特征缓存、真实 TRAIN 验收 | `bash scripts/prepare.sh` |
| 第一轮：seed3351，V0/V1/V1L/V2/V4/V5 | `bash scripts/train_matrix.sh pilot` |
| 完整矩阵：9组×3 seed×10 epoch | `bash scripts/train_matrix.sh full` |
| 秩和伙伴温度敏感性 | `bash scripts/sensitivity.sh` |
| 独立数值与流程测试 | `python -m unittest discover -s tests -v` |
| 完整命令帮助 | `python run.py --help` |

模型、损失、训练与评价分别在 `vgcpod/model.py`、`vgcpod/engine.py`、`vgcpod/metrics.py`。冻结 CLIP 源码随包提供，沿用历史提交和权重 SHA256。

本包实现 UT natural split 的第一轮实验。MIT-States/C-GQA 的模板、标签、分块和数据适配属于后续迁移阶段。
