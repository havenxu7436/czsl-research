# 14 · 冻结 CLIP 知识探针 r2

状态：七种固定配方的编码与评价已完成。

## 实验设置

- 模型：冻结 CLIP ViT-L/14，224px；optimizer steps=0。
- 配方：名称、正确 OE+AE、两份 OE+AE 置换、正确 AE、两份 AE 置换。
- TRAIN=22,998 张；VAL=3,214 张；VAL seen/unseen 分别为 877/2,337 张。
- GT-object 条件属性诊断；均值余弦分数温度 0.05。

## 流程

1. 检查数据划分、图片身份与提示文本。
2. 编码图像与七组固定文本。
3. 计算条件属性 micro、pair-macro、NLL 与逐类统计。

## 结果与文件

OE+AE 与 AE-only 的预设诊断判据均为 false。

- 执行代码（精简版未保留）
- 知识、配置与结果索引（精简版未保留）
- 执行命令（精简版未保留）
- results/recovered/czsl_clip_knowledge_probe_20261010_r2/result/metrics.csv（精简版未保留）
- results/recovered/czsl_clip_knowledge_probe_20261010_r2/result/summary.json（精简版未保留）
