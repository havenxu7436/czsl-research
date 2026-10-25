# 配置与策略来源

[trustcpf_stage_v1_10ep/configs](trustcpf_stage_v1_10ep/configs/) 保存完整配置，[trustcpf_stage_v1_10ep/assets](trustcpf_stage_v1_10ep/assets/) 保存知识资产。[experiment_inventory.json](experiment_inventory.json) 区分真实训练与共享权重评分，并逐项记录权重、指标文件及 SHA256。

26/26 阶段，15 个教师各追加 10 epoch，正式权重复用第一批 Uniform-best；81 个评分策略实例、972 行识别、744 行 Q5、486 份预测、165 行教师曲线。
