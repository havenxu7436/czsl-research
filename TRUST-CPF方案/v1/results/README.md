# 验证与结果

| 归档 | 内容 |
|---|---|
| [local-validation-20261021/](local-validation-20261021/) | 发布包原有13项测试、CUDA、CLIP tokenizer、公式和文档检查 |
| [repository-validation-20261021/](repository-validation-20261021/) | 仓库分区后的恢复、重复测试、链接与GitHub数学检查 |
| [deployment/](deployment/) | 完整服务器ZIP与SHA256 |
| [release_member_map.json](release_member_map.json) | 53成员→仓库位置→运行位置映射 |
| [trustcpf_clip_v1/](trustcpf_clip_v1/) | release manifest、provenance与CLIP BPE词表 |
| [package_support/](package_support/) | 原始README、方案、公式及命令的精确字节副本 |

13项测试覆盖概率归一化、零图、预算一/二阶性质、异方差梯度、配对干预与方差、概率平均、全方差/指标、五折覆盖、未测边隔离、分类梯度、CLIP视觉接口、exact AUC，以及五折到七组训练/噪声/report/恢复的合成流程。CUDA检查使用768维合成CLIP空间、96边、3次优化；概率归一化误差为5.96×10⁻⁸。Tokenizer检查316条prompt，全部在77-token上下文内。

当前正式UT训练完成数为0。这里保存实现和合成验证记录；正式识别指标及权重由服务器实验追加。

仓库新增的[文档检查器](../code/audits/check_repository_docs.py)复核链接、Python3.8语法、原始代码摘录与等价公式；[数学渲染检查器](../code/audits/check_github_math.cjs)使用KaTeX复核全部数学表达式。全仓库校验已通过，全仓库6787个规范文件的哈希检查通过。

GitHub网页排版已复核：[网页布局记录](repository-validation-20261021/github_layout.json)与[实际截图](repository-validation-20261021/github_layout_fixed.png)。修订方案的21个公式采用外置Markdown编号，避免浏览器MathML不支持的编号表格元素；53个行内公式、表格、数学内容及源码对应关系均已检查。
