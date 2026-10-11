# Bias=0 逐图诊断

只比较原始分数 bias=0 的 top-1。AUC、best-HM、best-seen/best-unseen 对应的曲线或 operating point 无法由这些 top-5 重建。

类别表同时记录样本量，低支持度类别不据此宣称稳定增益。各数据集失败原因见 issues.csv。
