# 配置与实验清单

[trustcpf_competition_v1_10ep/configs](trustcpf_competition_v1_10ep/configs/) 保存本批固定设置；[trustcpf_competition_v1_10ep/assets](trustcpf_competition_v1_10ep/assets/) 保存 96 条关系、标签和 CLIP 缓存指纹。

[逐 seed 实验清单](experiment_inventory.json) 标明选中 epoch、实际适配与 Uniform 复用。复用权重和历史已核对 SHA256。

代码默认 GPU 分配与实际执行记录分开保存。真实配置、运行身份和耗时见 [服务器 work](../results/raw_export/trustcpf_competition_v1_10ep/work/) 与 [setup 快照](../results/raw_export/trustcpf_competition_v1_10ep/setup.json)。
