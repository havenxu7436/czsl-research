# 分析文件与复现

**[完整分析报告](分析报告.md)** · HTML阅读版（精简版未保留） · [原始结果](../README.md)

主图位于 `figures/`，逐 seed 指标和类别细分位于 `tables/`，排版检查位于 `qa/`。原始科学数据与分析表均保持字节身份；发布报告已改为仓库相对链接和 GitHub 数学块。

## 从仓库复现

先拉取 LFS 并恢复到新的外部目录：


分析环境：Python 3.12.14、NumPy 2.3.5、PyTorch 2.5.1 CPU、Matplotlib 3.10.7。服务器训练环境仍以原始 protocol.json 为准。分析脚本只读取已有预测与权重，不进行训练；恢复后的原始文档还包含当次机器路径，复算命令使用当前环境的 `python` 即可。

历史完整归档包含三个 Python 分析脚本与页面检查脚本；精简版不保留这些脚本、环境路径或检查截图，数值结论保留在当前分析文档中。

配对统计先在同 seed 内相减，随后取均值和样本标准差；敏感性仅 seed3351。`AUC_native` 为原有限扫描，`AUC_exact` 为完整临界偏置扫描；固定偏置 HM 使用 VAL 偏置。`partner_effective_count` 为 exp(平均熵)，`gamma_variance_selected_epoch` 为日志中的池化方差。
