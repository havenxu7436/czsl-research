# TRUST-CPF Competition v1（10 epoch）

本批实现下一步 TRUST-Competition：在 CLIP ViT-L/14 缓存和强 CPF 上，比较三种关系效用监督能否改善组合识别。TRUST 的完整链条保留：TRAIN 内组合留出 → 关系开关收益 → 八维模型响应 → 27 参数效用预测 → 关系采样概率 → CPF 联合识别。

**整批为 5 方法 × 3 seed，共 15 组识别结果。** 默认物理 GPU 0/1/2 分别执行 seed 3351/3352/3353。三种监督共用 15 个新教师、相同风险图像和相同关系开关背景。最终 reference、知识库和图像特征沿用已完成的 Align。

| 方法 | 效用来源 |
| --- | --- |
| Uniform-0.5 | 所有关系概率为 0.5 |
| TRUST-U-NLL | 伪未见图像 NLL 下降 |
| TRUST-Balanced | 伪已见、伪未见各占一半的 NLL 下降 |
| TRUST-Competition | 平滑组合竞争面积提高 |
| Permuted-Competition | 类型内置换已选 Competition 概率 |

关键比较为 Competition−Balanced、Competition−Uniform、Competition−Permuted；同时保留 Balanced−U-NLL 和 Competition−U-NLL。

- [完整上传、服务器训练和下载命令](SERVER_COMMANDS.md)
- [公式、实现与核心指标](FORMULAS_AND_CODE.md)
- [默认配置](configs/default.json)

主要实现：`competition.py` 定义划分和风险公式；`competition_training.py` 完成教师、测量、拟合、扫描和 Q5 实际决策；`competition_workflow.py` 管理三 seed 队列、恢复和汇总。`engine.py` 与 `strong.py` 增加显式训练图像 ID，保证 G0 和关系教师都使用同一训练图像清单。分类器、原损失和 native/exact 识别指标沿用上一轮。

五组正式适配统一为 **10 epoch**，从同一 reference 开始，使用相同的首轮预热和十轮余弦学习率计划。在 epoch 0–10 中按 VAL CW AUC 选择最佳权重；所有任务完成第 10 轮后结束。每个 seed 重跑本批 Uniform；其他组若选中零强度，复用与之完全相同的本批 Uniform 结果。

新版目录为 `trustcpf_competition_v1_10ep`，与原三轮版本独立。`training_curves.csv` 汇总五组、三个 seed 从 epoch 0 到 10 的 165 行曲线；每轮保存损失、VAL CW/OW AUC、学习率及选中标记。日志 `val_auc` 为当前轮，`val_best` 为历史最佳值。

服务器继续使用 `vgcpod_v1` 的 Python 3.8、PyTorch 1.13.1 和 NumPy，不需要重新提取 CLIP 特征或安装新模型依赖。启动前运行交付测试；测试使用精简合成特征，全量 UT-Zappos 训练由服务器命令启动。

每 seed 包含 14 个阶段：导入、5 折教师与测量、拟合、Q5 决策、VAL 扫描、5 组适配。整批完成状态为 **42/42 阶段、15/15 教师、15/15 识别结果、3/3 Q5 审计**。报告有 360 行识别结果、96 行冻结扫描、312 行 Q5 策略／图 bank 结果。

本轮耗时主要来自 15 个新教师和完整关系测量，不能套用上一轮约 5.5 分钟的耗时。各 seed 顺序执行 5 折，每折 G0 10 epoch、关系头 10 epoch，完成一个测量背景立即打印进度。`costs.csv` 保存真实耗时。
