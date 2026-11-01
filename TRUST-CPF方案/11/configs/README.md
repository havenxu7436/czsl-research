# 配置、知识与完整队列

[165项任务及完成标记](experiment_queue.json) · [450项训练阶段](training_inventory.json) · [405份正式读出](recognition_inventory.json)。

发布配置、资产见[trustg0_three_v1_10ep](trustg0_three_v1_10ep/)。各数据集真实词表、metadata/split SHA256、情境分组、文本和关系缓存、五折划分均在[原始服务器导出](../results/raw_export/)。

队列由原jobs()与分阶段完成标记重建；逐项链接原日志和完成标记，不将其冒充下载包中不存在的全局队列日志。
