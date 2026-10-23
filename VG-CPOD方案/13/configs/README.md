# L组配置与实验清单

[最终配置](vgcpod_v1_roundL/configs/default.json) · [九组已完成清单](experiment_inventory.json) · [缓存指纹](vgcpod_v1_roundL/assets/expected_cache.json) · [资产来源](vgcpod_v1_roundL/assets/provenance.json)

L0=base、L1=knowledge、L2=shuffle；各seed3351/3352/3353、10 epochs、batch64。物理GPU1/2/3分别对应三个seed。Base参数36,864；Knowledge/Shuffle各110,593。CLIP、文本、Bo/Co/Au/Av固定；共同训练Bc/Cc/Wc。

[knowledge.pt](vgcpod_v1_roundL/assets/knowledge.pt)含OI12/OE36/AE48共96条关系；[shuffle.pt](vgcpod_v1_roundL/assets/shuffle.pt)固定seed7101，保持每类型source多重集、目标与度数。
