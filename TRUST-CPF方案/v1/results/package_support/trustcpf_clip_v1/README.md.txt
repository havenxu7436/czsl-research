# TRUST-CPF方案v1：CLIP版

本包实现冻结 CLIP ViT-L/14（224px）、对象条件属性预测、OI/OE/AE图残差、五折配对关系效用、异方差效用集成、风险与预算结构选择，以及UT-Zappos的CW/OW评价。

修订方案见 [TRUST-CPF_CLIP_修订实验方案v1.md](TRUST-CPF_CLIP_修订实验方案v1.md)，逐式代码见 [TRUST-CPF公式及代码v1.md](TRUST-CPF公式及代码v1.md)，命令见 [服务器实践命令.md](服务器实践命令.md)。

## 1. 文件

| 位置 | 内容 |
|---|---|
| `run.py` | 特征准备、教师、测量、效用、调参、结构、适配、噪声、汇总入口 |
| `trustcpf/model.py` | CLIP条件基底、图消息、三元残差、候选概率平均 |
| `trustcpf/utility.py` | 配对增益、视觉支持摘要、效用NLL、方差分解 |
| `trustcpf/stages.py` | 五折测量、三模型拟合、独立审计、Q4调参、最终结构与训练 |
| `trustcpf/engine.py` | 训练、AdamW、恢复、CW/OW校准与评价 |
| `trustcpf/graph.py` | 96条知识边、CLIP编码、预算求解、知识扰动 |
| `trustcpf/features.py`、`data.py` | 冻结CLIP缓存、UT数据与标签轴 |
| `trustcpf/metrics.py` | 原生有限偏置扫描、全临界点扫描 |
| `configs/default.json` | 正式默认配置 |
| `assets/knowledge/` | 既有UT reviewed-v2 OI/OE/AE原文 |
| `tests/`、`validation/` | 数学、编码接口、整条合成流程和恢复测试 |
| `vendor/clip/` | OpenAI CLIP源码与BPE；许可证同目录上层 |

## 2. 训练组

`pipeline`依次完成五折教师、效用测量、拟合/审计、Q4调参、五折标签最终拟合、参考教师及七组训练：`r0`、`capacity`、`dense`、`drop`、`free`、`mean`、`robust`。每组输出VAL/TEST、CW/OW、AUC、HM、NLL、ECE与图方差。

`shuffled`以单独work执行全流程，检查知识内容对应关系。`noise`执行训练后知识变化：错误描述附着/父类替换、关系删除；效用组用原有TRAIN证据重新预测结构。

## 3. 运行

服务器准备数据集、CLIP ViT-L/14权重和Python环境后：

```bash
source scripts/env.sh
bash scripts/prepare.sh
bash scripts/train_seed.sh 3351
```

首次正式训练建议先执行 `python -m unittest discover -s tests -v`。测试内容和实测状态保存在 `validation/`；测试用合成特征验证数学及调度，UT识别结果由服务器训练生成。

## 4. 缓存与恢复

图像缓存为 `g[N,768]`、`fa[N,16,768]`、`fo[N,12,768]`，FP16保存，读取为FP32并在适配器中单位化；文本保持FP32。特征准备逐分片复用；正式训练每epoch保存模型、AdamW、随机状态、epoch及最佳VAL权重，重复执行相同命令从last继续。测量文件按fold复用，效用MLP重跑时按固定种子重新拟合。

完整UT图像缓存约1.25GB（另有编码分片副本）；CLIP权重、输入图像独立存放。已有VG-CPOD同协议缓存可直接作为`--cache`输入，再执行`graph`生成本方法知识向量，无需重复编码图像。

## 5. 复用来源

CLIP提取、UT标签/图像路径映射和原生评价口径继承本机VG-CPOD v1经验证接口；条件分支、图融合、效用学习和结构训练为本包实现。来源身份与文件摘要见`provenance.json`，打包文件摘要见`release_manifest.json`。
