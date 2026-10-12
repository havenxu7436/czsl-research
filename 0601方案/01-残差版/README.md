# 01 · 残差版：A–H 与 sanity

状态：12 组×10 epoch、48 份评价已完成。

## 实验设置

- UT-Zappos；seed=3351；ViT B16、GloVe 300d；batch=64。
- 组别：A–H 与 H_no_pair/H_shuffle/H_random_prior/H_uniform_prior。
- G/H 在 native CPF composition 分数上加入 factorized head residual；λ=0.1、β=0.01、q=3。
- ViT、CPF 与新 head 联合优化；选模使用首次严格最大 VAL CW AUC。

## 流程

1. 准备缓存、逻辑 bank 与 12 组配置，执行 TRAIN 图像检查。
2. 各组从初始化训练 10 epoch。
3. 同一选中 checkpoint 评价 VAL/TEST CW/OW。

## 运行入口

变量与路径对应关系见 [环境说明](../../environment/README.md)。

```bash
REPLAY=/srv/czsl-reproduce-0601-01
python tools/reproduce.py materialize --line 0601方案 --stage 01 --dest "$REPLAY"
"$CPF_PY" "$REPLAY/reproduce_0601_20261010/prepare.py" --root "$PROJECT" --work "$WORK" --tier 1
"$CPF_PY" "$REPLAY/reproduce_0601_20261010/run_suite.py" --work "$WORK" \
  --gpus "${CZSL_GPUS:?Set physical GPU IDs}" --max-workers 1 \
  --deadline "${CZSL_DEADLINE:?Set a future ISO8601 deadline}" --export "$WORK/results.zip"
```

## 结果与文件

完整 H 的指标低于 A；主表与配对差分见 metrics.csv。

- [执行代码](code/reproduce_0601_20261010/)
- [知识、配置与结果索引](inputs.json)
- [执行命令](runbook.json)
- [results/extracted/summary/metrics.csv](<results/extracted/summary/metrics.csv>)
- [results/extracted/summary/summary.json](<results/extracted/summary/summary.json>)
