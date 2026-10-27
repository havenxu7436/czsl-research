# 配置、知识与运行清单

[trustcpf_abc_v1_10ep/configs](trustcpf_abc_v1_10ep/configs/) 为生产 10 epoch 设置；[trustcpf_abc_v1_10ep/assets](trustcpf_abc_v1_10ep/assets/) 为 96 条关系、标签与缓存指纹。三类 OI/OE/AE 的期望边数为 6/18/24，每图像混合 16 图。

[实验清单](experiment_inventory.json) 逐 seed、图库、策略记录共享权重、选中 epoch、别名和真实预测目录；共 234 个结果实例，不等于训练 234 个分类器。

本轮新训练三条 Uniform 轨迹，复用 15 个教师，效用预测器由九个教师联合拟合。Matched/B 为 27 参数，C 为 51 参数，岭系数 alpha=30；A 选择 eta=0.25，B/C 强度为 1。[外部输入](../external_inputs.json) · [实际状态](../results/raw_export/trustcpf_abc_v1_10ep/work/status.json)。
