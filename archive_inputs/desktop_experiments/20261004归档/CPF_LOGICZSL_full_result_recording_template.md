# CPF / LOGICZSL 论文完整复现：观测指标与数据记录总表

> 用途：训练、验证、测试结束后统一记录实验条件、逐 epoch 指标、最终论文指标和与论文的差值。
>
> 论文对比统一使用百分数，例如日志 `AUC=0.3605` 记为 `36.05`。原始小数也保留，便于排错。

---

## 1. 哪一组数才可以填入论文对照表

正式结果必须同时满足：

1. 使用论文对应的数据集和标准 compositional split；
2. 使用论文协议的 backbone、epoch、batch size、学习率和评估模式；
3. 每个 epoch 只在验证集评估；
4. 按**验证集 AUC**选择最佳 checkpoint；
5. 训练全部结束后，只加载该 checkpoint 在测试集评估；
6. Closed World（CW）和 Open World（OW）分别评估；
7. 最终论文表填写测试集的 `AUC / HM / Seen / Unseen`，不是训练集准确率，也不是最后一个 epoch 的验证结果。

以下结果不能混入正式论文对照：

- 1 epoch、batch 4 等冒烟测试；
- 使用测试集选 checkpoint 的结果；
- CW checkpoint 和 OW checkpoint 分别用测试集调出的结果；
- 数据 split、候选 pair 集合或 backbone 与论文不一致的结果；
- 重建知识文件得到的 Logic-Troika 结果冒充作者未公开知识文件的结果。

### 指标口径

| 字段 | 含义 | 是否是论文主表指标 | 记录要求 |
|---|---|---|---|
| AUC | 校准偏置扫描得到的 Seen-Unseen 曲线下面积 | 是 | 原始小数和百分数都记录 |
| HM / best_hm | Seen 与 Unseen 的最佳调和均值 | 是 | 直接记录 evaluator 输出 |
| Seen / best_seen | 已见组合准确率 | 是 | 直接记录 evaluator 输出 |
| Unseen / best_unseen | 未见组合准确率 | 是 | 直接记录 evaluator 输出 |
| Acc_Obj | 训练过程中对象分类准确率 | 否，诊断项 | 每个 epoch 末记录 |
| Acc_Attr | 训练过程中属性分类准确率 | 否，诊断项 | 每个 epoch 末记录 |
| Acc_Pair | 训练过程中组合分类准确率 | 否，诊断项 | 每个 epoch 末记录 |
| closed_attr_match | evaluator 输出的属性匹配率 | 否，诊断项 | 最终测试时记录 |
| closed_obj_match | evaluator 输出的对象匹配率 | 否，诊断项 | 最终测试时记录 |
| Loss | 训练损失 | 否，诊断项 | 每个 epoch 末记录 |
| max GPU memory | 最大显存占用 | 否，资源项 | 每次运行记录 |
| wall time | 总训练和评估耗时 | 否，资源项 | 每次运行记录 |

注意：不要用打印出来的 `best_seen` 和 `best_unseen` 自己重新计算 HM；不同实现中它们可能来自偏置扫描的不同位置。论文对比应直接采用同一 evaluator 输出的 `best_hm`。

---

## 2. 实验身份卡：每次正式运行先填一份

### 2.1 基本身份

| 字段 | 本次值 |
|---|---|
| Run ID | 例：`cpf_ut_zappos_cw_seed3351` |
| 项目 | CPF / Troika / Logic-Troika |
| 变体 | paper-aligned / repaired / reconstructed |
| 数据集 | MIT-States / UT-Zappos50K / C-GQA |
| Setting | CW / OW |
| Seed |  |
| 开始时间 |  |
| 结束时间 |  |
| 完成状态 | completed / failed / interrupted |
| 失败原因 | 成功则填 `—` |
| 运行目录 |  |
| train.log |  |
| eval.log |  |
| 最佳 checkpoint |  |
| checkpoint SHA256 |  |

### 2.2 代码、环境与硬件

| 字段 | 本次值 |
|---|---|
| Git commit / 源码快照 SHA256 |  |
| 本地修复说明或 patch |  |
| Python |  |
| PyTorch |  |
| Torchvision |  |
| CUDA runtime |  |
| NVIDIA driver |  |
| GPU 型号 |  |
| GPU 数量 |  |
| 精度 | FP32 / FP16 / AMP |
| 操作系统 |  |
| Conda environment 导出文件 |  |

### 2.3 训练协议

| 字段 | 本次值 |
|---|---|
| Backbone |  |
| 预训练权重路径 |  |
| 预训练权重 SHA256 |  |
| Epoch 数 |  |
| Optimizer |  |
| 初始 learning rate |  |
| Weight decay |  |
| Scheduler |  |
| Scheduler 参数 |  |
| 实际 batch size |  |
| Gradient accumulation |  |
| Effective batch size | `实际 batch × accumulation × GPU 数` |
| Test batch size |  |
| Num workers |  |
| 输入尺寸 |  |
| 训练数据增强 |  |
| 测试预处理 |  |
| 其他损失权重 |  |
| Context length | Troika/Logic-Troika 记录 |
| 验证起始 epoch |  |
| 验证频率 |  |
| 最优模型选择规则 | 应为 validation AUC maximum |

---

## 3. 数据集完整性记录

### 3.1 论文采用的标准规模

| 数据集 | 属性数 | 对象数 | 全部组合数 | Train 组合数 | Train 图像数 | Val Seen 组合 | Val Unseen 组合 | Val 图像数 | Test Seen 组合 | Test Unseen 组合 | Test 图像数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UT-Zappos50K | 16 | 12 | 192 | 83 | 22,998 | 15 | 15 | 3,214 | 18 | 18 | 2,914 |
| MIT-States | 115 | 245 | 28,175 | 1,262 | 30,338 | 300 | 300 | 10,420 | 400 | 400 | 12,995 |
| C-GQA | 413 | 674 | 278,362 | 5,592 | 26,920 | 1,252 | 1,040 | 7,280 | 888 | 923 | 5,098 |

### 3.2 本机审计结果

| 数据集 | 数据根目录 | split 文件 SHA256 | 实际属性数 | 实际对象数 | Train 图像 | Val 图像 | Test 图像 | 缺图数 | 重复/歧义图数 | 是否与论文一致 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| UT-Zappos50K |  |  |  |  |  |  |  |  |  |  |
| MIT-States |  |  |  |  |  |  |  |  |  |  |
| C-GQA |  |  |  |  |  |  |  |  |  |  |

Open World 还必须记录：

| 数据集 | 候选 pair 总数 | feasibility/compatibility 文件 | 文件 SHA256 | 筛选阈值 | 阈值选择依据 |
|---|---:|---|---|---:|---|
| UT-Zappos50K |  |  |  |  |  |
| MIT-States |  |  |  |  |  |
| C-GQA |  |  |  |  |  |

---

## 4. 每个 epoch 必须记录的观测值

一行代表一个 epoch。若日志只打印 iteration 级数据，至少记录该 epoch 最后一次或 epoch 平均值，并在“统计口径”中说明。

| Run ID | Epoch | Train Loss | Acc_Obj | Acc_Attr | Acc_Pair | Val AUC raw | Val AUC % | Val HM % | Val Seen % | Val Unseen % | LR group 1 | LR group 2 | LR group 3 | 最大显存 MB | Epoch 用时 | 是否刷新最佳 AUC | checkpoint 路径 | 统计口径/备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
|  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | 4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | 5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | 6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | 7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | 8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | 9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | 10 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | 11 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | 12 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | 13 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | 14 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | 15 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

CPF 正式训练只有 10 epoch，填写前 10 行；Troika/Logic-Troika 论文协议为 15 epoch，填写全部 15 行。

### 最优 epoch 摘要

| Run ID | 验证 AUC 最优 epoch | 最优 Val AUC % | 同次 Val HM % | 保存的 checkpoint | checkpoint SHA256 | 最后 epoch 是否等于最优 epoch |
|---|---:|---:|---:|---|---|---|
|  |  |  |  |  |  |  |

---

## 5. 最终 checkpoint 测试记录

同一个由验证集选出的 checkpoint 分别做 CW 和 OW 测试。不要在测试集上再次选模型。

| Run ID | Dataset | Setting | Seed | checkpoint | 选择 checkpoint 所用集合 | Test AUC raw | Test AUC % | Test HM raw | Test HM % | Test Seen raw | Test Seen % | Test Unseen raw | Test Unseen % | closed_attr_match % | closed_obj_match % | 测试耗时 | 评估是否成功 |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|  |  | CW |  |  | validation |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  | OW |  |  | validation |  |  |  |  |  |  |  |  |  |  |  |  |

换算示例：

```text
日志：AUC=0.2932, best_hm=0.4543, best_seen=0.5279, best_unseen=0.6552
表格：AUC=29.32, HM=45.43, Seen=52.79, Unseen=65.52
```

当前 CPF 冒烟结果只能写在独立记录中：

| 类型 | Dataset | Epoch | Batch | Setting | Test AUC % | Test HM % | Test Seen % | Test Unseen % | 可否对论文 |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| smoke | UT-Zappos | 1 | 4 | 当前 evaluator 设置 | 29.32 | 45.43 | 52.79 | 65.52 | 否 |

---

## 6. CPF 论文完整目标与复现填写表

差值定义：`Δ = 复现值 - 论文值`，单位均为百分点。

### 6.1 Closed World

| Dataset | Seed | Paper AUC | Repro AUC | ΔAUC | Paper HM | Repro HM | ΔHM | Paper Seen | Repro Seen | ΔSeen | Paper Unseen | Repro Unseen | ΔUnseen | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MIT-States | 3351 | 11.2 |  |  | 26.8 |  |  | 41.3 |  |  | 34.8 |  |  |  |
| UT-Zappos50K | 3351 | 41.4 |  |  | 55.7 |  |  | 66.4 |  |  | 71.1 |  |  |  |
| C-GQA | 3351 | 8.2 |  |  | 23.9 |  |  | 39.6 |  |  | 23.5 |  |  |  |

### 6.2 Open World

| Dataset | Seed | Paper AUC | Repro AUC | ΔAUC | Paper HM | Repro HM | ΔHM | Paper Seen | Repro Seen | ΔSeen | Paper Unseen | Repro Unseen | ΔUnseen | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MIT-States | 3351 | 4.4 |  |  | 15.1 |  |  | 40.8 |  |  | 14.4 |  |  |  |
| UT-Zappos50K | 3351 | 31.2 |  |  | 47.6 |  |  | 64.6 |  |  | 56.1 |  |  |  |
| C-GQA | 3351 | 2.1 |  |  | 9.5 |  |  | 38.4 |  |  | 6.8 |  |  |  |

---

## 7. LOGICZSL 论文完整目标与复现填写表

### 7.1 Closed World

| Dataset | Method | Seed | Paper AUC | Repro AUC | ΔAUC | Paper HM | Repro HM | ΔHM | Paper Seen | Repro Seen | ΔSeen | Paper Unseen | Repro Unseen | ΔUnseen | Knowledge 来源 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C-GQA | Troika | 0 | 12.4 |  |  | 29.4 |  |  | 41.0 |  |  | 35.7 |  |  | 无 |
| C-GQA | Logic-Troika | 0 | 15.3 |  |  | 33.3 |  |  | 44.4 |  |  | 39.4 |  |  | reconstructed / official |
| UT-Zappos50K | Troika | 0 | 41.7 |  |  | 54.6 |  |  | 66.8 |  |  | 73.8 |  |  | 无 |
| UT-Zappos50K | Logic-Troika | 0 | 45.8 |  |  | 57.8 |  |  | 69.6 |  |  | 74.9 |  |  | reconstructed / official |
| MIT-States | Troika | 0 | 22.1 |  |  | 39.3 |  |  | 49.0 |  |  | 53.0 |  |  | 无 |
| MIT-States | Logic-Troika | 0 | 23.4 |  |  | 40.5 |  |  | 50.8 |  |  | 53.9 |  |  | reconstructed / official |

### 7.2 Open World

| Dataset | Method | Seed | Paper AUC | Repro AUC | ΔAUC | Paper HM | Repro HM | ΔHM | Paper Seen | Repro Seen | ΔSeen | Paper Unseen | Repro Unseen | ΔUnseen | Knowledge 来源 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C-GQA | Troika | 0 | 2.7 |  |  | 10.9 |  |  | 40.8 |  |  | 7.9 |  |  | 无 |
| C-GQA | Logic-Troika | 0 | 3.4 |  |  | 12.6 |  |  | 43.7 |  |  | 9.3 |  |  | reconstructed / official |
| UT-Zappos50K | Troika | 0 | 33.0 |  |  | 47.8 |  |  | 66.4 |  |  | 61.2 |  |  | 无 |
| UT-Zappos50K | Logic-Troika | 0 | 36.2 |  |  | 50.8 |  |  | 69.6 |  |  | 63.7 |  |  | reconstructed / official |
| MIT-States | Troika | 0 | 7.2 |  |  | 20.1 |  |  | 48.8 |  |  | 18.7 |  |  | 无 |
| MIT-States | Logic-Troika | 0 | 8.7 |  |  | 22.4 |  |  | 50.7 |  |  | 21.4 |  |  | reconstructed / official |

### 7.3 Logic 相对 Troika 的配对增益

这张表比“只比较 Logic 的绝对值”更重要。两行必须来自相同数据、seed、backbone、训练配置和 evaluator。

| Dataset | Setting | Seed | Paper ΔAUC | Local Troika AUC | Local Logic AUC | Local ΔAUC | 与论文增益差 | Paper ΔHM | Local Troika HM | Local Logic HM | Local ΔHM | 与论文增益差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C-GQA | CW | 0 | +2.9 |  |  |  |  | +3.9 |  |  |  |  |
| UT-Zappos50K | CW | 0 | +4.1 |  |  |  |  | +3.2 |  |  |  |  |
| MIT-States | CW | 0 | +1.3 |  |  |  |  | +1.2 |  |  |  |  |
| C-GQA | OW | 0 | +0.7 |  |  |  |  | +1.7 |  |  |  |  |
| UT-Zappos50K | OW | 0 | +3.2 |  |  |  |  | +3.0 |  |  |  |  |
| MIT-States | OW | 0 | +1.5 |  |  |  |  | +2.3 |  |  |  |  |

由于作者没有公开实验所用的 OI/OE/AE 原始知识文件，使用 reconstructed knowledge 时，只能报告为“重建实验”，不能宣称严格复现 Logic-Troika 的官方绝对数值。

---

## 8. 多 seed 稳健性表

论文单个表格值可能对应作者固定 seed。为了证明结果不是偶然，建议至少追加 3 个 seed；CPF 的论文对齐主运行仍保留 seed 3351，额外 seed 单独报告。

### 8.1 每个 seed 原始结果

| Project | Method | Dataset | Setting | Seed | AUC | HM | Seen | Unseen | checkpoint SHA256 | 是否协议一致 |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
|  |  |  |  | 0 |  |  |  |  |  |  |
|  |  |  |  | 1 |  |  |  |  |  |  |
|  |  |  |  | 2 |  |  |  |  |  |  |

### 8.2 汇总

| Project | Method | Dataset | Setting | Seeds | AUC mean ± std | HM mean ± std | Seen mean ± std | Unseen mean ± std | Paper AUC | Paper HM |
|---|---|---|---|---|---|---|---|---|---:|---:|
|  |  |  |  | 0/1/2 |  |  |  |  |  |  |

标准差必须注明是样本标准差还是总体标准差；建议统一使用样本标准差 `ddof=1`。

---

## 9. 消融实验记录表

### 9.1 CPF

如果目标只是复现论文主表，可以先不跑全部消融；若要复现论文的完整实验章节，再按论文相应表格补充每个模块开关。

| Dataset | Setting | Seed | Base | CPF 模块/损失配置 | AUC | HM | Seen | Unseen | 相对 Base ΔAUC | 相对 Base ΔHM |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |  |  |  |

### 9.2 LOGICZSL

| Dataset | Setting | Seed | OI | OE | AE | Knowledge 版本 | AUC | HM | Seen | Unseen | 相对 Troika ΔAUC | 相对 Troika ΔHM |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| UT-Zappos50K | CW | 0 | off | off | off | none |  |  |  |  | 0 | 0 |
| UT-Zappos50K | CW | 0 | on | off | off | reconstructed-v1 |  |  |  |  |  |  |
| UT-Zappos50K | CW | 0 | off | on | off | reconstructed-v1 |  |  |  |  |  |  |
| UT-Zappos50K | CW | 0 | off | off | on | reconstructed-v1 |  |  |  |  |  |  |
| UT-Zappos50K | CW | 0 | on | on | on | reconstructed-v1 |  |  |  |  |  |  |

---

## 10. 运行资源和异常记录

| Run ID | 总训练时长 | CW 测试时长 | OW 测试时长 | 最大 GPU 显存 MB | CPU 峰值 | RAM 峰值 | OOM 次数 | 中断/重启次数 | NaN/Inf | 数据加载错误 | 其他异常 |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |

若出现 OOM 后改变 batch、AMP 或梯度累积，必须创建新的 Run ID，不能覆盖原实验记录。

---

## 11. 复现结论判定

以下是本项目内部诊断标准，不是论文官方标准：

| 结论 | 条件 |
|---|---|
| 高度接近 | 协议一致，AUC 和 HM 绝对差都不超过 1.0 个百分点 |
| 基本复现 | 协议一致，AUC 和 HM 绝对差大多不超过 2.0 个百分点，趋势一致 |
| 部分复现 | 绝对差 2–5 个百分点，或存在已披露的硬件/精度/配置差异 |
| 未复现 | 绝对差超过 5 个百分点，或 split、候选集合、checkpoint 选择方式不一致 |
| 不可直接比较 | 冒烟测试、epoch 不同、不同 backbone、非官方知识或测试集参与选模 |

每个最终实验写一句结论：

```text
本次 [方法/数据集/Setting] 使用 [协议摘要]，测试 AUC/HM 为 [x/y]，
相对论文差值为 [Δx/Δy] 个百分点；依据上述标准判定为 [结论]。
主要不可控差异为 [硬件、知识文件、精度策略等]。
```

---

## 12. 每次跑完后的实际抄数顺序

1. 确认日志末尾有 `Done` 或正常退出码；
2. 记录总 epoch、batch、seed、数据集、CW/OW 和精度；
3. 从每个 epoch 的 validation 行填写第 4 节；
4. 找到验证 AUC 最大的 epoch；
5. 核对保存并加载的是该 epoch 的 checkpoint；
6. 从最终 test 行抄 `AUC / best_hm / best_seen / best_unseen`；
7. 原始小数乘 100 后填入论文对照表；
8. 记录 checkpoint 与配置文件 SHA256；
9. 计算 `复现值 - 论文值`；
10. 填写复现等级和差异原因；
11. 保留完整日志、配置、环境导出和 checkpoint，不要只保留截图。

CPF 日志可先这样抽取关键行：

```bash
grep -E 'Epoch:|Validation set|Val/Test Epoch|Saved new best|Loading validation-best|Test set|Max GPU|New Best|Done:' \
  "$FORMAL_ROOT/train.log"
```

计算 checkpoint 校验和：

```bash
sha256sum "$CPF_CKPT"
```

保存环境和 GPU 信息：

```bash
conda env export --no-builds > "$FORMAL_ROOT/conda_environment.yml"
python -m pip freeze > "$FORMAL_ROOT/pip_freeze.txt"
nvidia-smi -q > "$FORMAL_ROOT/nvidia_smi.txt"
cp config/ut-zappos.yml "$FORMAL_ROOT/config_used.yml"
```

---

## 13. 最终论文复现交付物清单

每个正式 Run ID 至少应有：

```text
RUN_ID/
├── train.log
├── eval_cw.log
├── eval_ow.log
├── config_used.yml
├── command.txt
├── conda_environment.yml
├── pip_freeze.txt
├── nvidia_smi.txt
├── code_commit_or_snapshot.txt
├── data_manifest.txt
├── checkpoint_sha256.txt
├── epoch_metrics.csv
├── final_metrics.csv
└── checkpoints/
```

最终报告至少包含：

- CPF：3 个数据集 × CW/OW，共 6 个主结果行；
- Troika：3 个数据集 × CW/OW，共 6 个主结果行；
- Logic-Troika：3 个数据集 × CW/OW，共 6 个主结果行；
- 每行 4 个论文指标：AUC、HM、Seen、Unseen；
- 每行与论文的 4 个绝对差值；
- Logic-Troika 相对 Troika 的配对增益；
- 数据、环境、配置、checkpoint 和异常记录；
- reconstructed knowledge 的明确声明。

