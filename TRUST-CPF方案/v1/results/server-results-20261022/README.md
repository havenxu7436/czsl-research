# TRUST-CPF v1 · 2026-10-22 服务器完整结果

主矩阵 7 组 × 3 seed，共 **21 组最终识别模型**；另有一次知识打乱完整流程、两项效用消融审计及 600 次噪声评估。53 个发布成员复用既有源码归档，381 个新增服务器成员按原目录保存；434 个成员均可通过映射核对 SHA256。

## 主结果

以下为三个训练 seed 的 TEST native AUC 均值 ± 样本标准差，单位为点。

| 模型 | TEST CW AUC | TEST OW AUC |
| --- | --- | --- |
| `r0` | 2.020 ± 2.232 | 2.010 ± 2.215 |
| `capacity` | 13.902 ± 0.885 | 11.240 ± 1.065 |
| `dense` | 11.169 ± 2.488 | 9.214 ± 2.309 |
| `drop` | 10.915 ± 1.545 | 9.047 ± 1.257 |
| `free` | 10.896 ± 1.519 | 9.011 ± 1.319 |
| `mean` | 10.080 ± 2.102 | 8.249 ± 1.692 |
| `robust` | 9.901 ± 1.890 | 8.172 ± 1.515 |

![三 seed 主结果](../../analysis/server-results-20261022/confirmed/figures/01_main_auc.png)

`robust` 相对 `capacity` 平均低 4.000 CW 点、3.069 OW 点；相对 `drop` 平均低 1.014 CW 点、0.875 OW 点。完整解释及下一轮三组实验见 [完整分析](../../analysis/server-results-20261022/README.md)。

## 数据位置

| 位置 | 内容 |
| --- | --- |
| [raw_export/trustcpf_clip_v1/work](raw_export/trustcpf_clip_v1/work/) | 五折教师、效用测量／模型／审计、参考模型、21 主模型及 shuffled 的全部权重与历史 |
| [原始任务日志](raw_export/trustcpf_clip_v1/logs/) | 三 seed、知识打乱、噪声及消融完整日志 |
| [原始总表](raw_export/trustcpf_clip_v1/work/summary.csv) | 服务器 report 导出的全部结果 |
| [原始压缩包](archives/trustcpf_clip_v1_results_20261022.tar.gz) | 下载的完整 TAR.GZ，保持原字节，Git LFS 保存 |
| [压缩包 SHA256](archives/trustcpf_clip_v1_results_20261022.tar.gz.sha256) | 原包校验值 |
| [434 个成员映射](server_member_map.json) | 原始运行路径 → 仓库规范路径 → 大小／SHA256 |
| [归档完整性核对](validation/source_archive_integrity.json) | gzip CRC、中文路径修复、434 文件逐字节身份记录 |
| [Markdown 与公式检查](validation/markdown_checks.json) | 归档文档的公式、相对链接及表格结构检查记录 |
| [页面渲染记录](validation/local_browser_layout.json) | 完整分析的11个块公式、52个行内公式及4张统计图均正常，未见横向溢出 |
| [仓库总校验](validation/repository_verify.json) | 文件哈希、Markdown、Python语法及12条既有路线校验通过 |
| [全部运行清单](../../configs/server-results-20261022/run_inventory.json) | 22 个终端识别模型的选中 epoch、指标与 checkpoint |
| [完整派生分析与 CSV](../../analysis/server-results-20261022/confirmed/) | 主结果、配对差值、效用审计、同样本消融及噪声长表 |
| [图像与运行截图](images/README.md) | 15 张原图及其 SHA256；完整数据统计图另存 analysis |
| [原分析文档字节](package_support/) | 发布到仓库前的 Markdown 原文 |

## 噪声数据恢复

原噪声队列完成 600 次评估。后续快速检查覆盖了 `robust/noise_mismatch.json`，现存 JSON 总计 529 次评估；完整日志保留 600 条 OW AUC。恢复结果与全部现存对应 OW 数值逐条一致，最大误差为 0。

完整 OW 比较使用 [600 条恢复记录](../../analysis/server-results-20261022/confirmed/noise_ow_recovered.csv)。robust 错配的完整 CW 数值未记录在日志中，原文件的 4 条记录保持原样，不与其他完整 CW 曲线混合汇总。噪声仅使用训练 seed3351；五个扰动 seed 不作为五次模型训练。

## 校验与重算

```bash
python tools/register_trustcpf_v1.py --check
python tools/register_trustcpf_results_20261022.py --check
python tools/reproduce.py verify
```

命令在仓库根目录执行。分析重算见 [REPRODUCE.md](../../analysis/server-results-20261022/REPRODUCE.md)。UT 数据集原图、原始 metadata/split 与 CLIP 预训练权重继续按 [外部输入索引](../../external_inputs.json)提供；本批完整保留服务器导出的实验数据、权重、日志、统计图和讨论截图。
