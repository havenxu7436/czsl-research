# TRUST-CPF Strong v1

完整 TRUST v1 在既有强 CPF 基底上的实验版本，日期 2026-10-23。

本包沿用本机已经验证的 G0：冻结 CLIP ViT-L/14 全局特征、条件概率分解和静态 CP32。包含 seed3351/3352/3353 的 G0 权重。TRUST 保留类型消息、三元残差、五折配对效用、异方差集成、预算选图、图概率平均及原 v1 全套对照。

- 方法、公式与实验范围：[PLAN.md](PLAN.md)
- 上传、三卡训练、下载命令：[SERVER_COMMANDS.md](SERVER_COMMANDS.md)
- 与源码一致的核心公式及实现：[FORMULAS_AND_CODE.md](FORMULAS_AND_CODE.md)
- 验证记录：`validation/tests.log`、`validation/full_queue_synthetic.json`

完整队列使用 GPU1/2/3 分别执行 seed3351/3352/3353；共 114 个阶段，生成 24 组主/打乱知识模型结果、6 组固定权重读出结果、12 份效用审计和 1,800 条噪声评价。队列阶段完成后保存标记；训练按 epoch 恢复，效用测量按折恢复，噪声按条件恢复。

本机已验证实际 G0 权重的无图输出等价性，并在合成数据上跑通完整队列。正式 UT 数据位于服务器；本包不包含新的正式识别结果。2026-10-23 的免密 SSH 检查返回认证失败，服务器训练由文档中的启动命令执行。

依赖：Python 3.8+、PyTorch 1.13+、NumPy 1.24+；可直接使用现有 `vgcpod_v1` 环境。无需重新编码图像或下载 CLIP。
