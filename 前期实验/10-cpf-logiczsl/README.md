# 10 · CPF＋LogiCZSL B/F

状态：四组×10 epoch 已完成。

## 实验设置

- UT-Zappos；seed=3351；A_ref、B、E_ref、F。
- 逻辑损失：OI/OE/AE，q=3，β=0.01；F 使用文本缓存与 λ=0.1。
- CPF 任务损失为 L_comp + 0.4 L_obj + 0.6 L_attr。

## 流程

1. 构建逻辑 bank，生成四组配置。
2. 执行 TRAIN 图像梯度检查与 10 epoch 训练。
3. 按 VAL CW AUC 选模并评价 VAL CW/OW。

## 运行入口

变量与路径对应关系见 [环境说明](../../environment/README.md)。

```bash
REPLAY=/srv/czsl-reproduce-10
python tools/reproduce.py materialize --line 前期实验 --stage 10 --dest "$REPLAY"
"$CPF_PY" "$REPLAY/logiczsl_cpf_bf_20261008/prepare.py" \
  --root "$PROJECT" --work "$WORK" --seed 3351 --allow-reconstructed
for GROUP in A_ref B E_ref F; do
  "$CPF_PY" "$REPLAY/logiczsl_cpf_bf_20261008/run.py" \
    --work "$WORK" --seed 3351 --variant "$GROUP" --gpu "${CZSL_GPU:?Set physical GPU ID}"
done
"$CPF_PY" "$REPLAY/logiczsl_cpf_bf_20261008/summarize.py" --work "$WORK" --out "$WORK/summary"
```

## 结果与文件

B/F 相对匹配参考组的指标增量见原始运行记录。

- [执行代码](code/logiczsl_cpf_bf_20261008/)
- [知识、配置与结果索引](inputs.json)
- [执行命令](runbook.json)
