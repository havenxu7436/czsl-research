# v2 交付与验证

[部署ZIP](deployment/trustcpf_conditional_v2.zip) · [ZIP SHA256](deployment/trustcpf_conditional_v2.zip.sha256) · [成员映射](release_member_map.json) · [发布清单](trustcpf_conditional_v2/MANIFEST.json)

三个G0初始权重和原始protocol保存在[initial/](trustcpf_conditional_v2/initial/)，对应seed3351/3352/3353的原选中epoch8/8/9。

[本机验证](local-validation-20261022/release_validation.json)与[测试日志](local-validation-20261022/tests.log)记录5项测试：真实G0初始输出一致、消息公式与冻结梯度、知识打乱不变量、VAL校准传递、九组流水线及中断续训。流水线使用合成特征；正式UT-Zappos九组结果待服务器运行。

仓库归档另检查40个发布成员的SHA256、部署ZIP、逐字节恢复、Python3.8语法、Shell换行、Markdown链接及数学渲染；记录见[仓库验证](repository-validation-20261022/checks.json)。

原始部署文档保存在`package_support/`，仓库顶层阅读文档采用GitHub公式排版。训练结果后续存入独立的服务器归档目录。
