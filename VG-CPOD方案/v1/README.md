# v1 · VG-CPOD

**[完整公式及核心代码](VG-CPOD公式及代码v1.md)** · [数学与实验设计](数学与实验设计_v1.md) 

状态：三 seed 主矩阵 27 组与敏感性 6 组均完成 10 epochs，共形成两组冻结基线、140 份评价和 66 份 best/last 权重。精简档案保留[正式结果](results/ut-zap50k-3seeds-20261018/README.md)与[完整分析](results/ut-zap50k-3seeds-20261018/analysis/分析报告.md)，不保留原始权重和预测文件。

## 实验设置

- 冻结CLIP ViT-L/14图像与文本双编码器，224px，768维共同空间。
- UT-Zappos natural split；全局图像与类别查询特征统一缓存。
- V0、V1、V1L、V2–V7；seed3351/3352/3353共27次正式训练，每组10 epoch；另完成6组seed3351敏感性训练。
- 首次严格最大VAL CW `AUC_native`选模；同一权重评价VAL/TEST CW/OW，分别保存`AUC_exact`。

## 结果与校验

- [三 seed 正式结果与逐运行指标](results/ut-zap50k-3seeds-20261018/README.md)
- [七图分析报告](results/ut-zap50k-3seeds-20261018/analysis/分析报告.md)与[分析口径](results/ut-zap50k-3seeds-20261018/analysis/README.md)
- [来源及许可](来源与许可.md)

名称 CLIP、链式零样本、D0 和完整 VG-CPOD 分别保持原评分定义；合成数据检查不计入 UT 正式精度。
