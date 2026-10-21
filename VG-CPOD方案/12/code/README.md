# 12 · 代码与修订

原运行源码位于[vgcpod_v1_roundK](vgcpod_v1_roundK/)。配置、资产和验证按平行目录保存，[成员映射](../package_members.json)供materialize恢复原包。相对[11 J组](../../11/code/README.md)的完整差异见[source_changes.patch](source_changes.patch)。

## 1. Wd独立学习率组

[engine.py](vgcpod_v1_roundK/vgcpod/engine.py)的`make_optimizer`在配置含`dynamic_readout_lr_multiplier`时，将Wd从普通CP组移出：

```python
if separate_dynamic:
    scale = cp_scale * multiplier
    groups.append({'params': [model.Wd], 'name': 'dynamic_cp',
                   'lr_scale': scale, 'lr': cfg['lr'] * scale})
```

adapter组为Bo/Co/Bc/Cc，CP组为Au/Av/Bvis/Wc，dynamic_cp只含Wd。三个组的lr_scale分别为1、0.3、0.09或0.03。`set_learning_rates`将同一余弦日程乘组倍率。无新配置键时保持原J1参数分组与顺序。

本轮[model.py](vgcpod_v1_roundK/vgcpod/model.py)与[mean_gate_eval.py](vgcpod_v1_roundK/mean_gate_eval.py)和J组逐字一致，继续使用separate读出、原损失及TRAIN均值评估。可训练参数147,456，新增参数为0。

## 2. 从真实Adam状态核验更新

新增[readout_diagnostics.py](vgcpod_v1_roundK/readout_diagnostics.py)：

1. 按模型与优化器参数顺序，将checkpoint中的参数ID映射为Wc/Wd。
2. 读取step、exp_avg、exp_avg_sq、β₁/β₂、ε、lr及lr_scale，做Adam偏差修正。
3. 分别输出矩阵范数、一阶／二阶矩、归一化方向和学习率乘自适应方向的范数。
4. 对native选中、exact选中和末轮checkpoint分别记录，并报告动态／静态比值。

该诊断中的更新范数对应最后一步自适应项；weight decay单独执行。诊断模型初始化保留调用方Torch RNG。

## 3. 调度与输出

| 文件 | 内容 |
|---|---|
| [round_k.py](vgcpod_v1_roundK/round_k.py) | 每卡K1→K2；复制J组TRAIN均值；完成十二项均值评估和历史配对 |
| [tmux脚本](vgcpod_v1_roundK/scripts/start_round_k_tmux.sh) | 当前Python、会话、日志与退出码；从完整epoch续训 |
| [K1/K2模板](../configs/round_k/) | Wd倍率0.3/0.1 |
| [历史参考](../configs/references/pre_k_complete.json) | 38项已有运行快照，含J1 |
| [ranking_diagnostics.py](vgcpod_v1_roundK/ranking_diagnostics.py) | K1/K2−J1及K1−K2等逐图配对 |
| [优化器总表](../results/raw_export/roundK_20261021/readout_optimizer_summary.csv) | J1/K1/K2实际更新量 |

## 4. 验证与结果复算

- [61项发布测试](../results/package_validation/K/validation/unit_tests.txt)、[摘要](../results/package_validation/K/validation/unit_test_summary.json)、[CPU六训练十二评估](../results/package_validation/K/validation/result.json)。
- [test_round_k.py](vgcpod_v1_roundK/tests/test_round_k.py)：分组成员与倍率、首末学习率、相同梯度下的AdamW更新、配置合法性及J1兼容性。
- 原J1配置在新旧训练器中的参数、AdamW和Torch RNG逐元素相同；CPU流程重复启动跳过已完成训练和评估。
- [正式结果复算](../analysis/recompute.py)、[原分析脚本](original_analysis/)、[仓库验证](../analysis/repository_validation.json)、[服务器源码身份](../results/server_source_identity.json)。

CLIP许可见[CLIP_LICENSE](vgcpod_v1_roundK/vendor/CLIP_LICENSE)。发布验证与正式服务器训练分别记录；公式及超参数见[主说明](../README.md)，结果解释见[末项数据分析](../analysis/README.md)。
