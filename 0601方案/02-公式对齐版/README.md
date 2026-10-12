# 02 · 公式对齐版 r2

状态：seed 3351 的14组×30 epoch、56份评价已完成。

## 实验设置

- UT-Zappos；seed=3351；ViT B16、GloVe 300d；batch=64。
- 组别：A–H、J0/Jtext、H_no_pair/H_shuffle/H_random_prior/H_uniform_prior。
- G/H：score = z_obj[:, :, None] + z_attr_obj_K + λ·prior；GT-object CE 读取 z_attr_obj_K。
- 逻辑 grounding 使用 object softmax、条件属性 logsumexp、pair/description sigmoid；β=0.01、q=3。
- 训练与推理共享评分接口；首次严格最大 VAL CW AUC 选模。

## 流程

1. 准备单 seed 配置并执行 TRAIN 图像公式与梯度检查。
2. 14 组各训练 30 epoch。
3. 同一选中 checkpoint 评价 VAL/TEST CW/OW，导出原始分数与日志。

## 运行入口

变量与路径对应关系见 [环境说明](../../environment/README.md)。

```bash
REPLAY=/srv/czsl-reproduce-0601-02
python tools/reproduce.py materialize --line 0601方案 --stage 02 --dest "$REPLAY"
"$CPF_PY" "$REPLAY/faithful_0601_20261011/prepare.py" --root "$PROJECT" --work "$WORK" \
  --epochs 30 --seeds 3351
"$CPF_PY" "$REPLAY/faithful_0601_20261011/run_suite.py" --work "$WORK" \
  --gpus "${CZSL_GPUS:?Set physical GPU IDs}" --max-workers 1 --seeds 3351 --export "$WORK/results.zip"
```

只读结果校验：

```bash
python 0601方案/02-公式对齐版/code/epoch_review_20261012/verify_result.py
```

## 结果与文件

14 组最佳 VAL CW checkpoint 位于 epoch 2–6；56 份指标由保存分数复算，最大原始数值差为 0。

- [执行代码](code/faithful_0601_20261011/)
- [知识、配置与结果索引](inputs.json)
- [执行命令](runbook.json)
- [results/seed3351-30epoch-20261012/README.md](<results/seed3351-30epoch-20261012/README.md>)
