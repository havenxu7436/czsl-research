# 02 · 公式对齐版 r2

> **[完整公式及代码（0–12 章）](复现公式及代码.md)**
>
> 公式对齐版 r2 的数学定义、核心实现和源码索引。

状态：seed 3351、3352、3353 的 14 组矩阵已完成，共 42 次训练、168 份评价。

## 实验设置

- UT-Zappos；seed=3351/3352/3353；ViT B16、GloVe 300d；batch=64。
- 组别：A–H、J0/Jtext、H_no_pair/H_shuffle/H_random_prior/H_uniform_prior。
- G/H：score = z_obj[:, :, None] + z_attr_obj_K + λ·prior；GT-object CE 读取 z_attr_obj_K。
- 逻辑 grounding 使用 object softmax、条件属性 logsumexp、pair/description sigmoid；β=0.01、q=3。
- 训练与推理共享评分接口；首次严格最大 VAL CW AUC 选模。

## 流程

1. 按 seed 与训练预算准备配置，并执行 TRAIN 图像公式与梯度检查。
2. seed 3351 的 14 组各训练 30 epoch；seed 3352、3353 的 28 组各训练 10 epoch。
3. 同一选中 checkpoint 评价 VAL/TEST CW/OW，导出原始分数与日志。

## 运行入口

变量与路径对应关系见 [环境说明](../../environment/README.md)。

```bash
REPLAY=/srv/czsl-reproduce-0601-02
python tools/reproduce.py materialize --line 0601方案 --stage 02 --dest "$REPLAY"
WORK3351="$WORK/seed3351-30epoch"
"$CPF_PY" "$REPLAY/faithful_0601_20261011/prepare.py" --root "$PROJECT" --work "$WORK3351" \
  --epochs 30 --seeds 3351
"$CPF_PY" "$REPLAY/faithful_0601_20261011/run_suite.py" --work "$WORK3351" \
  --gpus "${CZSL_GPUS:?Set physical GPU IDs}" --max-workers 1 --seeds 3351 --export "$WORK3351/results.zip"
```

seed 3352、3353 使用单独的 10 epoch 运行目录：

```bash
WORK2="$WORK/seeds3352-3353-10epoch"
"$CPF_PY" "$REPLAY/faithful_0601_20261011/prepare.py" --root "$PROJECT" --work "$WORK2" \
  --epochs 10 --seeds 3352,3353
"$CPF_PY" "$REPLAY/faithful_0601_20261011/run_suite.py" --work "$WORK2" \
  --gpus "${CZSL_GPUS:?Set physical GPU IDs}" --max-workers 1 --seeds 3352,3353 --export "$WORK2/results.zip"
```

只读结果校验：

```bash
python 0601方案/02-公式对齐版/code/epoch_review_20261012/verify_result.py
python 0601方案/02-公式对齐版/code/epoch_review_20261012/verify_additional_results.py
```

## 结果与文件

| seed | 每组实际 epoch | 训练组数 | 评价份数 | 结果入口 |
|---|---:|---:|---:|---|
| 3351 | 30 | 14 | 56 | [指标与训练曲线](results/seed3351-30epoch-20261012/README.md) |
| 3352 | 10 | 14 | 56 | [完整结果](results/seeds3352-3353-10epoch-20261012/README.md#seed-3352) |
| 3353 | 10 | 14 | 56 | [完整结果](results/seeds3352-3353-10epoch-20261012/README.md#seed-3353) |

三批结果均按首次严格最大 VAL CW AUC 选模，同一权重评价 VAL/TEST CW/OW。seed 3351 的最终选中权重位于 epoch 2–6，仍保留其实际 30 epoch 的完整记录。

- [执行代码](code/faithful_0601_20261011/)
- [知识、配置与结果索引](inputs.json)
- [执行命令](runbook.json)
- [results/seed3351-30epoch-20261012/README.md](<results/seed3351-30epoch-20261012/README.md>)
- [seed 3352、3353：完整结果与校验](results/seeds3352-3353-10epoch-20261012/README.md)

## 数据分析（2026-10-13）

[数据分析与原方案问题覆盖评估](数据分析_原方案问题覆盖评估_20261013.md)：三个 seed 的配对结果、CW/OW 差异、sanity 归因、原始分数诊断，以及原设计已完成和仍待解决的问题。实际训练预算为 30/10/10 轮，统计使用核验后的共同前 10 轮选模视角。
