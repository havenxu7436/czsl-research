# VG-CPOD v1 结果分析复现

报告位于上级目录 `VG-CPOD_v1_实验结果分析_20261018.md`，HTML 为同内容的便携阅读版。分析输入是上级目录的 `server_results_20261018`，所有结果写入本目录，不覆盖原始实验数据。

## 执行顺序

在已安装 NumPy、PyTorch、Matplotlib 的 Python 环境中运行：

```powershell
python analysis_20261018/analyze_results.py
python analysis_20261018/detail_breakdown.py
python analysis_20261018/build_report.py
```

本次使用 Python 3.12.14、NumPy 2.3.5、CPU PyTorch 2.5.1、Matplotlib 3.10.7。归档服务器运行环境记录在各组 `protocol.json` 中；本地只读取已保存结果与权重，不重新训练。

本机现有依赖的运行命令：

```powershell
Set-Location 'C:\Users\llenovo\Documents\ChatGPT\VG-CPOD'
$env:PYTHONPATH='C:\Users\llenovo\Documents\ChatGPT\CZSL\tmp\vgcpod_test_deps'
$env:PYTHONIOENCODING='utf-8'
$env:MPLCONFIGDIR="$PWD\analysis_20261018\mpl_cache"
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$vgAnalysisPython='C:\Users\llenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $vgAnalysisPython analysis_20261018/analyze_results.py
& $vgAnalysisPython analysis_20261018/detail_breakdown.py
& $vgAnalysisPython analysis_20261018/build_report.py
```

## 输出与口径

`audit.json` 保存完整性、哈希、checkpoint 与逐图重算核对结果；`analysis_data.json` 保存主矩阵、配对差值、机制与预测诊断；`detail_data.json` 保存属性贡献与混淆细分。

`tables/` 为 UTF-8 BOM CSV，`figures/` 为七张 PNG。均值与标准差基于 seed3351/3352/3353，标准差使用 `ddof=1`。配对差值先在同 seed 内相减。敏感性只有 seed3351。AUC 单位为曲线面积 × 100，准确率与 HM 以百分数计，NLL 使用自然对数。

`test_pair_contributions.csv` 的 `contribution_total_cond_points` 是该组合对整体 GT-object 属性准确率变化的贡献，等于图像数占比乘条件准确率变化；`cw_pair_acc_calibrated` 是使用各自 VAL CW 偏置时的组合准确率，两个量分别解释条件分支与组合预测。

`mechanisms.csv` 中 `delta_reference_gamma1_norm_mean` 仅为把门控设为 1 时的算子基准范数，不是实际逐图修正幅度。`partner_effective_count` 为 exp(平均熵)，不是逐图有效数的算术均值。`gamma_variance_selected_epoch` 是日志中跨图像、物体、模式的池化方差。

`per_pair_metrics.csv` 的 CW/OW 各使用对应候选集和 VAL 偏置。OW 的 91 个额外候选是合法的开放世界候选；其胜出率按 TEST 未见图像作分母，并在未见候选偏置趋于正无穷时计算。

归档缺少用于新前向推理的九个图像特征数组；服务器已有缓存可直接用于报告提出的固定权重诊断。备份内的 Python 源码按发布哈希验证后，仅导入 `vgcpod/metrics.py` 的确定性评价函数。

## HTML 阅读版

在项目根目录，使用 Pandoc 生成含嵌入图片的 HTML：

```powershell
& 'C:\Users\llenovo\anaconda3\Scripts\pandoc.exe' '--standalone' '--self-contained' '--mathml' '--from=markdown+tex_math_dollars+pipe_tables' '--to=html5' '--css=analysis_20261018/report.css' '--metadata=pagetitle:VG-CPOD v1 实验结果分析' 'VG-CPOD_v1_实验结果分析_20261018.md' '-o' 'VG-CPOD_v1_实验结果分析_20261018.html'
```
