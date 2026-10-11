# EKR v2：从第 3 轮续训至第 10 轮

本包对应 2026-10-08 的训练预算回溯决定。回到共享值对齐 v2 的四项实验：`V_name_align`、`V_clean_align`、`V_shuffle_align`、`V_clean_fixed`。seed、共同底座、描述、打乱映射、初始化、模型结构、LR、optimizer、损失与选择规则保持原设置；新增训练为每组第 4–10 轮，共 28 轮。原 3 轮结果与所有旧 gate 保留。

默认只读来源：

```text
/var/tmp/czsl_xhw_name_ekr_20261007
/var/tmp/czsl_xhw_ekr_v2_align_20261007
```

默认新输出：

```text
/var/tmp/czsl_xhw_ekr_v2_align_budget10_20261008
```

## 执行

使用 `/data/xhw/miniconda3/envs/cpf/bin/python`，不依赖当前 shell 是否激活 conda。必须有原四组的完整 `last_epoch.pth`、选中 checkpoint、旧结果和旧源码。prepare 要求系统盘至少 5 GiB 可用空间。

第一次执行：

```bash
bash /tmp/ekr_v2_budget10_20261008/run.sh all
```

`all` 会依次准备独立输出、导入旧断点、续训并汇总。网络断开后可以在原 tmux 会话查看；如果进程已经退出，在相同环境下再次运行同一条命令，会验证已导入状态并跳过已完成的组。训练中断时从最后一轮已提交的完整断点继续。只有发生在一个 epoch 中途的未提交工作才需要重做。

也可分步执行：

```bash
bash /tmp/ekr_v2_budget10_20261008/run.sh prepare
bash /tmp/ekr_v2_budget10_20261008/run.sh import
bash /tmp/ekr_v2_budget10_20261008/run.sh resume
bash /tmp/ekr_v2_budget10_20261008/run.sh report
```

不要在已完成分步 prepare 后再次要求新建输出。这里的 `all` 会识别已有准备目录；`prepare` 本身仍保持禁止覆盖。

## 导入与审计

1. 对照本地已经审阅的旧 v2 manifest 检查 167 个文件、旧协议和旧 release。模型 `core.py` 与旧版逐字节一致。
2. 回放共同底座；其来源固定为 A_unit 的旧选中权重与 N_name 的第 3 轮选中权重。两者持续冻结。
3. 检查四份第 3 轮断点的变体、协议、选中 checkpoint、三轮记录、1080 个 optimizer steps、超参数及五类 RNG 状态。
4. 用真实 GPU 回放每份第 3 轮 router，核对原 CW 全量分数、图像、候选与标签。最大分数容差为 1e-5，原生指标容差为 0.05 pp；任何不一致都会停止。
5. 保留旧 epoch 1–3 评估文件；只把新副本绑定到新协议。模型参数、optimizer 和 RNG 内容不重新初始化，导入后的第一轮是 epoch 4。旧源文件保持只读。
6. 完整导入四组后原子提交新 runs 目录，并保存不可变导入证据 SHA。再次运行会核对这些证据。
7. 每轮提交完整断点并保存 VAL CW 全量分数。最终以 epoch 0–10 的第一个严格 CW AUC 最大值选权重，然后执行 VAL CW/OW。第 0 轮是关闭新分支的共同底座。

本地已进行 CPU 小模型的导入与中断集成检查；这不是服务器真实 GPU 验证。实际 GPU 回放由 import 阶段执行。真实 GPU 训练尚未由助手启动。

旧原生 cuDNN benchmark 设置保持原样，可能产生数值非确定性。CPU 中断恢复测试中的精确一致不代表服务器 GPU 训练逐位相同；实际回放按上面的容差检查。

## 结果含义

沿用旧 v2 主判据：正确描述对共同底座、名称对照、打乱对照的 CW AUC 都至少 +0.2 pp，保护条件保持原值。正确描述对 fixed 消融至少 +0.1 pp 是另外记录的对齐收益判据。

这次在见过 3 轮结果后增加预算，属于探索性预算复核；不能当作独立确认，也不会改写旧的 false。即使 10 轮未过，结论也只限于当前实现、设置、seed 和预算。10 轮不自动代表收敛；单 seed 不代表跨 seed 稳定。若有优势，应另立固定的多 seed 配对确认协议。

## 输出

```text
/var/tmp/czsl_xhw_ekr_v2_align_budget10_20261008/budget_import.json
/var/tmp/czsl_xhw_ekr_v2_align_budget10_20261008/reports/decision.json
/var/tmp/czsl_xhw_ekr_v2_align_budget10_20261008/reports/results.csv
/var/tmp/czsl_xhw_ekr_v2_align_budget10_20261008/reports/epoch_curves.csv
/var/tmp/czsl_xhw_ekr_v2_align_budget10_20261008/reports/budget3_vs_budget10.csv
/var/tmp/czsl_xhw_ekr_v2_align_budget10_20261008/reports/budget_scope.json
/var/tmp/czsl_xhw_ekr_v2_align_budget10_20261008_review.zip
```

审阅 ZIP 含分数与记录，不含训练权重；服务器 checkpoint 继续保留。训练阶段估算 2 小时 55 分钟，另加检查、回放、最终评估与打包，建议预留 3–3.5 小时，以实际计时为准。

如果提示旧断点缺失、来源 SHA 不符或回放失败，保留输出与日志，不能用 best checkpoint 替代 last_epoch 并假称从第 3 轮恢复。输出磁盘空间不足时释放合适的缓存或另设符合前缀要求的新版本目录；不要覆盖旧 runs。
