# 07 · 共享值对齐 EKR v2

状态：四组×3 epoch 已完成。

## 实验设置

- 共同底座：冻结 A_unit 与 N_name。
- 组别：V_name_align、V_clean_align、V_shuffle_align、V_clean_fixed；seed=0。
- 共享低秩 value 映射 rank=32；主对照 48,034 个可训练参数，fixed 对照 28,834 个。

## 流程

1. 准备共同底座、bank 与四组配置。
2. 执行分支训练，保存 epoch 0 与训练轮次的 VAL CW 指标。
3. 按 VAL CW AUC 选模，评价 CW/OW 并汇总。

## 运行入口

变量与路径对应关系见 [环境说明](../../environment/README.md)。

```bash
REPLAY=/srv/czsl-reproduce-07
python tools/reproduce.py materialize --line 前期实验 --stage 07 --dest "$REPLAY"
bash "$REPLAY/ekr_v2_anchor_alignment_20261007/run.sh" prepare
bash "$REPLAY/ekr_v2_anchor_alignment_20261007/run.sh" train
bash "$REPLAY/ekr_v2_anchor_alignment_20261007/run.sh" report
```

## 结果与文件

V_clean_align 选中 epoch 0；V_clean_fixed 相对共同底座有小幅增量。

- [执行代码](code/ekr_v2_anchor_alignment_20261007/)
- [知识、配置与结果索引](inputs.json)
- [执行命令](runbook.json)
- [results/artifacts/reports/results.csv](<results/artifacts/reports/results.csv>)
