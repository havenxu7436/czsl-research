# 版本演进与原始来源

| 顺序 | 文档 | 定位 |
|---|---|---|
| 1 | [USG-CPF 3.0 可迁移语义关系效用：原始idea](source/USG-CPF_3.0_可迁移语义关系效用_研究方案.md.txt) | 关系效用与可迁移预测的初始研究设想 |
| 2 | [USG-CPF 3.0 完整实验方案v1](source/USG-CPF_3.0_完整实验方案_v1.md.txt) | 第一版逐步数学方案，使用ViT-B/16＋GloVe |
| 3 | [TRUST-CPF CLIP修订实验方案v1](../TRUST-CPF_CLIP_修订实验方案v1.md) | 当前执行方案，冻结ViT-L/14共同空间，低秩图消息、配对效用与风险预算 |
| 4 | [公式及核心代码v1](../TRUST-CPF公式及代码v1.md) | 当前公式到真实源码的逐式映射 |

前两份以`.md.txt`保留原始Markdown字节和公式。当前代码和训练配置以CLIP修订版为准。原始文档SHA256见[source_manifest.json](source_manifest.json)；CLIP包内文档的原始副本保存在[package_support](../results/package_support/)。

公式代码的编排参照第三路线[VG-CPOD公式及代码v1](../../../VG-CPOD方案/v1/VG-CPOD公式及代码v1.md)。
