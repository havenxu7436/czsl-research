# 代码

完整执行源码位于trustcpf_clip_v1（精简版未保留），共包含CLI、核心模块、服务器脚本、测试和CLIP源码。

| 模块 | 职责 |
|---|---|
| run.py（精简版未保留） | 各独立阶段与pipeline入口 |
| trustcpf/（精简版未保留） | 数据、缓存、知识图、模型、配对效用、训练、评价与报告 |
| scripts/（精简版未保留） | prepare、单seed训练、噪声队列、公式文档提取及打包 |
| tests/test_trustcpf.py（精简版未保留） | 数学恒等式、梯度、划分、全流程与恢复测试 |
| vendor/clip/（精简版未保留） | 冻结CLIP双编码器与tokenizer实现 |
| requirements.txt（精简版未保留） | Python依赖 |

本目录记录历史实现的模块划分；执行源码与资产不在精简版中。[公式及核心代码](../TRUST-CPF公式及代码v1.md)中的摘录来自同一版本。
