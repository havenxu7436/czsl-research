# v1 结果与验证

| 内容 | 范围 |
|---|---|
| [三seed正式结果](ut-zap50k-3seeds-20261018/README.md) | 27组主矩阵、6组敏感性、两组冻结基线；140份评价、66份权重、原始预测与日志 |
| [完整分析与图像](ut-zap50k-3seeds-20261018/analysis/分析报告.md) | 主矩阵、同seed消融、未见组合与Suede细分、门控/伙伴/正交诊断、敏感性及下一轮顺序 |
| [原始服务器导出](ut-zap50k-3seeds-20261018/archives/VG-CPOD_v1_results_20261018.tar.gz)与[成员映射](ut-zap50k-3seeds-20261018/export_member_map.json) | 579个导出成员逐项保留并校验；大文件使用Git LFS |
| [CPU检查](local-validation-20261018/result.json)与[日志](local-validation-20261018/tests_cpu.log) | 本机12项数值/合成流程测试；包括9组前向、训练、选模和评价 |
| [验证说明](验证记录.md) | 运行环境与检查范围 |
| [GitHub网页版检查](repository-validation-20261018/github_browser_rendering.json) | 修复51处不兼容宏；4个网页的324个公式全部生成MathML，渲染错误为0 |
| [公式审计](formula-audit-20261018/) | 桌面公式文档的梯度核对、渲染报告和46个独立公式的截图 |
| [部署包](deployment/VG-CPOD方案v1.zip) | 36个冻结文件，恢复时逐项校验；ZIP/BPE/图片由Git LFS存储 |
| [成员映射](release_member_map.json) | 原运行路径到code/configs/results分区的对应 |
| [原始文档](package_support/) | 字节保留的桌面与冻结包文档；文件扩展为`.md.txt` |

2026-10-18新增服务器正式结果及分析。原始图像特征数组和数据集图像未包含在服务器下载包中；已下载的权重、预测、曲线、元数据及图像全部归档。公式审计快照中的本机路径为当次生成位置，仓库源码入口以版本页和可移植公式文档为准。
