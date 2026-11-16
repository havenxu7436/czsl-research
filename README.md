# CZSL 研究：精简实验档案

本仓库保留 **方案演进、最终公式、核心代码与完整分析结论**，不再把训练数据、checkpoint、逐样本导出和运行日志当作 GitHub 仓库内容。这样可直接阅读每条研究路线的证据链，同时避免用 Git/LFS 承担数十 GiB 的实验产物。

## 最终保留路线

| 路线 | 重点版本 | 为什么保留 | 入口 |
|---|---|---|---|
| 前期验证 | 01–15 的 Markdown 证据链 | 记录从 CPF 基线到冻结 CLIP 探针的关键否定/正向证据 | [索引](前期实验/README.md) · [汇总](前期实验/RESULTS_SUMMARY.md) |
| 0601 | 02 公式对齐版 + v2 统一修订版 | 02 有三 seed 完整复现；v2 是最新统一公式实现 | [路线说明](0601方案/README.md) |
| VG-CPOD | 12 动态读出 + 13 知识残差 | 12 完成更新尺度诊断；13 是最后的三 seed 语义对照 | [路线说明](VG-CPOD方案/README.md) |
| TRUST-CPF | 10 完整 ABC + 11 三数据集独立流水线 | 10 完成机制拆解；11 是覆盖 UT/MIT/C-GQA 的最终泛化实验 | [路线说明](TRUST-CPF方案/README.md) |

## 阅读方式

1. 先看各路线 README 的版本选择与结论。
2. 进入重点版本的 `analysis/README.md`、`results/README.md` 或公式文档查看完整表格。
3. 最终版本保留可审阅的 `code/` 与 `configs/`；旧版本只保留分析性 Markdown。
4. CW/OW 分别表示 closed-world/open-world。训练组按对应 VAL 规则选模，选中权重用于结果评价。

## 数据可用性

当前精简版不包含数据集、知识缓存、模型权重、checkpoint、日志、逐样本预测和压缩包。Git 历史仅保留当前核心文件的演进，已移除 LFS 指针与跟踪规则；代码、文档与小型结果文件通过普通 Git 上传。GitHub 后台已有的 LFS 对象不会因此自动返还配额，详情见 [CLEANUP_POLICY.md](CLEANUP_POLICY.md)。

环境版本与外部输入约定仍见 [environment/README.md](environment/README.md)。

## 上传代码与结果

清理历史后，请重新克隆仓库；旧工作目录中的未提交代码和结果先另存，再按文件复制到新克隆中。不要合并旧分支、推送旧标签或使用 `git push --all`，否则可能重新引入已删除的大文件历史。

```bash
git clone https://github.com/havenxu7436/czsl-research.git
cd czsl-research
git config --local core.hooksPath .githooks
```

安装 Python 3 后，提交与推送时会自动检查 LFS 指针、原始实验产物和超过 10 MiB 的文件。保留源码及必要配置、核心 Markdown、CSV/JSON/TSV 结果表和小型结果图；建议将结果放在对应路线的 `results/` 下。权重、数据集、缓存、运行日志与压缩包不进入本仓库。

```bash
git add <代码或结果文件路径>
python tools/check_repository.py
git commit -m "更新代码与实验结果"
git push origin main
```

本仓库不需要执行 `git lfs track` 或 `git lfs push`。上述限制是仓库的体积策略，不是 GitHub 单文件上限。
