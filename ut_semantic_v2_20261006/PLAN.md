# 本次实施状态与固定边界

已完成训练图板与84句的助手审核，生成独立v2字典，冻结类别顺序、描述数量、GloVe协议、原生训练配置、描述错配映射及门槛。

现在按 RUN_UT_V2.md 执行 prepare.sh 与 run_paired.sh 即可。服务器实际cache、bank和12配置由原构建器生成并校验；包内不放合成词向量冒充真实缓存。默认独立工作区避免继续占用曾满盘的/data。

这次预算为一次知识修订、四版本×三个训练seed×10epoch。完成12组后先审计日志/权重/评估/配对样本，再按原gate作决定。不给成功概率；不修改原gate、不弱化阈值、不以验证结果继续反复改词。

通过时，enter_ekr_pilot.sh 用新的同seed C0权重、独立实验目录运行R0/R3/R5/R7 frozen seed0×3epoch。先GPU验证零增量复现、off恢复、基线冻结和梯度更新。pilot不自动扩到其他seed或其他数据集。

未通过时默认停止本条知识修订向正式EKR推进；USG随机语义图路线仍独立。原v1所有结果和失败证据保留。

助手审核不等于人工专家批准，也不等于视觉能鉴定材料来源/性能。review_status.csv中的approved只表示助手完成本轮可用性审阅；manifest和84条记录明示human_reviewed=false。
