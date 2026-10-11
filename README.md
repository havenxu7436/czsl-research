# CZSL 私人研究档案

截至 **2026-10-11（北京时间）**。本仓库保存 CPF / KCC / EKR / LogiCZSL 相关研究的代码版本、知识资产、实验配置、结果、复核和研究决策。原始文件保持原位置和原字节；通过研究阶段索引组织阅读。

## 从哪里开始

1. [研究阶段索引](docs/EXPERIMENTS.md)：按研究依赖顺序，查阅每次修改的动机、实际执行、证据、结论和后续决策。
2. [截至 10 月 10 日的完整实验总档案](docs/history_20261010.md)：包含逐步骤公式、完整结果表、训练/诊断/复用的区别和负结果。
3. [知识来源、恢复与构建档案](docs/knowledge_history_20261010.md)：OI/OE/AE、名称、模板 prior、知识缓存的来源与处理。
4. [两条 0601 路线对照](docs/ROUTES.md)：区分已训练的 residual 版与主公式对齐版。
5. [统一指标索引](docs/METRICS.md)及 [metrics_index.csv](provenance/metrics_index.csv)：保留阶段、数据集、seed、VAL/TEST、CW/OW和原始文件身份，不跨协议拼成排行榜。

## 两条路线分别保留

| 路线 | 代码入口 | 结果/状态入口 |
|---|---|---|
| `residual-0601`：旧 native CPF + residual 实现 | [reproduce_0601_20261010](reproduce_0601_20261010/README.md) | [12 组实际训练结果](output/0601_priority_results_20261011/extracted/summary/0601_实际训练与汇总.md)、[结果与公式复核](docs/residual_review_20261011.md) |
| `formula-aligned-0601`：主评分/损失公式对齐 r2 | [faithful_0601_20261011](faithful_0601_20261011/README.md) | [完整代码与配置归档](archive_inputs/formula_aligned_bundle/README.md)、[历史 CPU 验证记录](archive_inputs/formula_aligned_bundle/verification/desktop_bundle_verification.json)；截至归档未收集到本版本正式服务器训练结果 |

两个版本均保留。主公式对齐版不继承旧 residual 版的研究成绩；CPU 测试通过也不计为研究实验完成。原方案、接口选择、剩余项和具体公式见路线对照。

## 文件布局

| 位置 | 内容 |
|---|---|
| `docs/` | 新增的可在 GitHub 浏览的研究导航、阶段卡和相对链接报告副本 |
| 各带日期的根目录 | 已有代码包、模型/审计实现、运行脚本和知识文件，原样保留 |
| `output/` | 原始实验结果、CSV/JSON、逐轮日志、预测数组、审计、失败记录和发布 ZIP |
| `archive_inputs/desktop_experiments/` | 从桌面早期研究归档补入的科学资料；按原目录结构保存 |
| `archive_inputs/formula_aligned_bundle/` | 桌面上已交付的主公式对齐版本完整包，包含其原始 manifest |
| `archive_inputs/referenced_sources/` | 原报告引用的其他已存在来源，逐文件映射 |
| `provenance/` | SHA256 清单、外部来源映射、未上传的大资源清单、指标索引及归档验证 |
| `archive_tools/` | 归档构建、只读验证和使用既有 Git 登录凭据的私人仓库操作工具 |

提交历史是 **2026-10-11 按研究依赖顺序进行的回溯归档提交**，不是当时实验当天的开发提交。每项实际时间、seed、预算、是否续训/回放以协议和原始结果为准。

## 下载与后续使用

本仓库的结果压缩包、分数数组和图像使用 Git LFS。第一次下载：

```powershell
git lfs install
git clone -c core.longpaths=true https://github.com/havenxu7436/czsl-research.git
cd czsl-research
git lfs pull
python archive_tools/archive.py verify
```

`git lfs pull` 下载真实证据文件；仅下载 GitHub 的普通源码 ZIP 可能得到 LFS 指针。数据集原图全集、GloVe 大文件、预训练模型和大型训练 checkpoint 保留原存储位置，仓库清单明确标注。不能据此声称本仓库可无外部输入从零复现实验。

新增实验请遵循 [实验记录与 Git 使用约定](docs/WORKFLOW.md)。不要把旧报告中的启动命令当作当前待执行任务；这次归档没有连接服务器或启动训练。

本机的大输出、依赖和归档缓存已迁至 D 盘，原 C 盘路径保持可用，详见 [本机存储位置说明](docs/STORAGE.md)。这不影响其他电脑克隆仓库。
