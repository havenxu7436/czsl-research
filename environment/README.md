# 实验环境记录

## 基础软件

| 项目 | 归档实验环境 |
|---|---|
| 操作系统 | Linux |
| Python | 3.8.18 |
| PyTorch / CUDA | 1.13.1 + cu117 |
| torchvision | 0.14.1 + cu117 |
| NumPy / SciPy | 1.24.4 / 1.10.1 |
| 训练 GPU | RTX 3090 |
| CLIP | OpenAI CLIP ViT-L/14 |

## 数据与模型输入

| 路线 | 外部输入 |
|---|---|
| 前期实验 / 0601 | UT-Zappos、MIT-States、C-GQA 的图像与原始 split/metadata；GloVe 300d；对应 ViT 权重与前序 checkpoint |
| VG-CPOD | UT-Zappos natural split、CLIP ViT-L/14 权重与冻结图像/文本特征 |
| TRUST-CPF | UT-Zappos、MIT-States、C-GQA；CLIP ViT-L/14；各数据集独立冻结特征与 CPF/G₀ 基底 |

数据集图像、预训练模型、冻结特征、知识缓存和 checkpoint 不属于精简仓库。各重点版本的配置文件保留数据集、seed、epoch、选模规则和超参数定义。

## 评价口径

- 0601 方案按对应 VAL 规则选择权重，并以同一权重评价 VAL/TEST CW/OW。
- VG-CPOD 使用首次严格最大 VAL CW AUC 选模，同时记录 native/exact AUC。
- TRUST-CPF 的正式比较在同 seed、同图库和同一冻结分类器条件下配对；版本 11 对三个数据集分别拟合流水线。

精简仓库用于审阅公式、实现与结果，不包含完整训练环境快照或原始实验工作目录。
