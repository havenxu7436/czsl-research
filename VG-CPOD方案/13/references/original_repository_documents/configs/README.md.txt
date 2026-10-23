# v2 配置与知识

[最终训练配置](trustcpf_conditional_v2/configs/default.json) · [9组实验矩阵](experiment_matrix.json) · [G0缓存指纹](trustcpf_conditional_v2/assets/expected_cache.json) · [资产来源](trustcpf_conditional_v2/assets/provenance.json)

[knowledge.pt](trustcpf_conditional_v2/assets/knowledge.pt)保存OI12、OE36、AE48共96条知识；[shuffle.pt](trustcpf_conditional_v2/assets/shuffle.pt)采用seed7101，96条附件均改变，保持类型、目标、度数及每类型source多重集。两文件通过Git LFS存储。

v2冻结Au/Av与Bo/Co，降低已有条件头的续训学习率；早期方向中的自由Au/Av及较高学习率已由[修订方案](../REVISED_PLAN.md)替换。
