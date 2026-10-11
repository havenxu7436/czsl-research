# 16 · residual-0601：统一A–H与sanity

状态：已完成12组×10轮、48份评价。

## 修改与结论

以旧native CPF＋residual接口补齐八主组与四独立对照，统一VAL选模与VAL/TEST评价。

核心工程范围完成，完整H低于A；先验、文本、逻辑与head影响按配对和边界解释。

## 复现顺序

1. 按 [统一环境与外部输入说明](../../environment/README.md) 准备数据、GloVe、预训练模型。
2. 在仓库根目录校验并生成独立运行目录；此步骤不训练、不覆盖已有目录。

```bash
python tools/reproduce.py verify
python tools/reproduce.py materialize --stage 16 --dest /srv/czsl-reproduce-16
```

3. 运行目录保留历史包内相对位置及哈希。基础工程位于 `project/`，执行包位于运行目录根部。按统一环境说明处理冻结路径和外部输入；路径迁移不能绕过原哈希检查。
4. 优先使用包的 `--root`、`--work` 或环境变量配置路径，先预检/prepare，再执行本阶段训练或诊断。检查结果的 dataset、seed、VAL/TEST、CW/OW、选模与复用身份。
5. [inputs.json](inputs.json) 登记本阶段资产、配置与结果；[runbook.json](runbook.json) 提供去重的实验命令模板及来源；模板路径变量和包内默认路径均须核对。

## 命令变量

以下命令从独立运行目录读取文件。先设置匹配的 Python 与已验证的工程根目录；
历史包内部的默认路径仍须按环境说明处理，不能只替换命令中的变量。

```bash
REPLAY=/srv/czsl-reproduce-16
export CPF_PY="${CZSL_PYTHON:?Set the matching CPF Python executable}"
export PROJECT="${CZSL_PROJECT_ROOT:?Set a verified compatible project root}"
```

## 关键入口

以下是 Linux 执行顺序；先完成环境说明中的历史路径映射和原包预检。`CZSL_GPUS` 必须设为已授权的物理 GPU 列表，`CZSL_DEADLINE` 必须是未来的带时区 ISO8601 时间。新 work 必须与已有实验分开。

```bash
REPLAY=/srv/czsl-reproduce-16
PROJECT="${CZSL_PROJECT_ROOT:?Set a verified compatible project root}"
WORK=/srv/czsl-runs/residual-0601-new
"$CPF_PY" "$REPLAY/reproduce_0601_20261010/prepare.py" --root "$PROJECT" --work "$WORK" --tier 1
"$CPF_PY" "$REPLAY/reproduce_0601_20261010/run_suite.py" --work "$WORK" \
  --gpus "${CZSL_GPUS:?Set authorized physical GPU IDs}" --max-workers 1 \
  --deadline "${CZSL_DEADLINE:?Set a future ISO8601 deadline}" \
  --export "$WORK/results.zip"
```

先核对生成的 `protocol.json`，再启动第二条命令。tier 1 是 12 组×10 轮；新复跑不导入旧主表。

执行包：[reproduce_0601_20261010](<code/reproduce_0601_20261010>)。

运行时入口：`/srv/czsl-reproduce-16/reproduce_0601_20261010/`。其原始 README/哈希校验所需说明会从不可变归档标签恢复到运行目录，不在当前仓库重复堆放。

前序阶段：[README.md](<../15-train-knowledge-audit/README.md>)。阶段依赖不等同于每一步都必须重新训练；续训/回放需要对应原权重。

知识版本：[cache_manifest.json](<../../knowledge/ut-zap50k/definitions/legacy-v1/cache_manifest.json>)、[cache_manifest__b7a7e056f2a7.json](<../../knowledge/ut-zap50k/definitions/legacy-v1/cache_manifest__b7a7e056f2a7.json>)、[cache_manifest__c5b68542cd99.json](<../../knowledge/ut-zap50k/definitions/legacy-v1/cache_manifest__c5b68542cd99.json>)、[cache_manifest__ee036a3a5c5f.json](<../../knowledge/ut-zap50k/definitions/legacy-v1/cache_manifest__ee036a3a5c5f.json>)、[labels.json](<../../knowledge/ut-zap50k/definitions/legacy-v1/labels.json>)、[bank.pt](<../../knowledge/ut-zap50k/logic/reconstructed-v1/bank.pt>)、[bank__cb584f863880.pt](<../../knowledge/ut-zap50k/logic/reconstructed-v1/bank__cb584f863880.pt>)、[coverage.json](<../../knowledge/ut-zap50k/logic/reconstructed-v1/coverage.json>)；完整清单见 inputs.json。

原始结果：[metrics.csv](<results/extracted/summary/metrics.csv>)、[summary.json](<results/extracted/summary/summary.json>)。

配置文件：[A.yml](<configs/extracted/configs/A.yml>)、[B.yml](<configs/extracted/configs/B.yml>)、[C.yml](<configs/extracted/configs/C.yml>)、[D.yml](<configs/extracted/configs/D.yml>)；已生成的运行配置和 protocol 也在结果中，完整文件见 inputs.json。

**复跑注意：**历史 residual 队列有 2026-10-11 截止时间。新复跑必须给队列指定未来的授权截止时间，不能沿用过去的默认日期。性能表只属于这条 residual 路线。

<details>
<summary>模型定义、对照与协议边界</summary>

### residual 主矩阵与独立对照

#### 执行范围

本阶段范围为 UT-Zappos、seed 3351、12 组各 10 epoch；多数据集、多 seed 与描述删除扩展不计入本阶段完成量。

| 优先级 | 实验 | 新完整训练数量 | 为什么先做 |
|---|---|---:|---|
| P0 / tier 0 | 同协议 A–H，先发 G/H/C/D，再发 A/B/E/F | 8 | 补真正缺失的 Full Model，重新建立严谨主表 |
| P1 / tier 1 | H_no_pair、H_shuffle、H_random_prior、H_uniform_prior | 4 | 原方案五项 sanity：主 G/H 已经是 template-only，另外四项从头训练 |
| P2 / tier 2 | H_noise10/30/50 | 3 | 仅 Text/Feature 描述删除敏感性；明确不是全部知识噪声实验 |
| 留后 | 描述 1/3/5、OI 深度 1/2/3、MIT/C-GQA 全 A–H、多 seed | 不计入本包 | 新知识/层级定义与大规模扩展不能冒充已完成 |

默认 tier 1：12 组，均为 10 epoch。tier 2：15 组。建议先采用 tier 1，确保核心主表与必要对照优先完成。如果确有多张空闲卡且时间充足，可在启动前选择 tier 2；不能运行中编辑已经冻结的 protocol。

没有可靠 LLM feasibility 时按原文 §4.5 使用 text-only，不伪造 LLM-only/mixed 分数。主 G/H 的 pair 资产完全是全笛卡尔积模板；template-only 被覆盖，但由于没有 pair-level LLM 资产，不存在 LLM-vs-template 的对比结论。

完成标准按计划执行与汇总；负增益也保留。

#### 具体模型

- 原生 CPF/GloVe 300d、ViT B16、UT seed 3351、batch 64、10 epoch、原 split/评价器。
- 原有配置 `finetune_backbone=True`，保留 ViT、CPF 的联合优化策略。不是冻结缓存特征，不是 25% 小头混合，也不是 CLIP 蒸馏。
- G/H 的 §6.1 head：四个 300→512 投影和 pair MLP，共 771,073 个新参数。实际 CPF 是两条 B×300 共享视觉向量；通过矩阵乘得到 B×N / B×M，再合成 B×N×M 分数，不展开大特征张量。
- 适配选择：保留 native composition 分数，把 factorized K 分数作为 residual 加入，而非删掉原 composition 模块。训练新 head 的分数除以原温度 0.05；推理与原 legacy 余弦尺度一致。
- G/H 的属性辅助损失是 `CE(native_attr_logits + K_delta[GT_object], attr)`。训练组合 CE 使用实际 TRAIN 候选，不用 argmax object 筛候选。
- H 的属性逻辑分数来自实际 TRAIN pair scores 按属性聚合的 logsumexp；对象和描述节点沿用真实 CPF 视觉分支与重建 bank。B/F 保留已有原语属性 adapter。因此 H−F 同时涉及 head、条件 CE 和 adapter 变化，不声称这是只增加一个 MLP 的单变量比较。
- 校准 λ=0.1；逻辑 β=0.01、q=3，β 只乘一次；A–F feature=false 时保持审计过的旧行为。
- feature 参数在独立 RNG 上初始化，不改变 CPF 基础参数初始化和后续采样 RNG；原生 Python/NumPy/Torch seed 对齐，cudnn benchmark=false，不承诺 GPU bitwise determinism。
- H_no_pair 只关闭 pair MLP 的消费，四个 primitive projection 仍训练。pair MLP 留在结构中但梯度为 None，不把其参数当作活跃参数。

知识仍有两个记录在案的来源：原 v1 Text/prior/pair 缓存；已审计 B/F 的重建 logic bank。没有改成另一套新知识，也不混入旧批次结果。A–H 本次全部重新训练。

#### 控制实验的含义

- H_shuffle：两个固定无自映射的 permutation，交换 OE/AE 描述归属；原语名称、OI、pair 模板和 prior 保持原值。Text/Feature 和逻辑 OE/AE 使用相同交换索引。
- H_random_prior：固定 seed 603354，随机 [12,16] prior，centered 后零均值/单位总体标准差，尺度与真实 prior 一致；λ 不变。
- H_uniform_prior：常数矩阵经 centered 变换为零，独立从头训练，回答训练中先验的贡献。
- 队列结束时还在真实 H VAL OW 分数上检验常数 prior 与关闭 prior 的算术恒等；这不是额外模型 forward/训练，报告明确区分。
- P2 噪声：按 OE/AE 各自固定打乱的描述 slot 删除，10/30/50 为目标比例，有整数取整。只重聚合 Text/Feature primitive K；logic/prior/pair 模板不变。具体删除索引在 control manifest，不能据此声称所有知识树的鲁棒性都测过。

#### 时间和资源

已有真实 F 日志从 2026-10-08 17:16:06 UTC 到 18:25:59 UTC，约 70 分钟（含训练与 VAL）。新增 full head 和 TEST/检查有额外开销。建议按 80–110 分钟/组做排期，不保证这个范围。

- P0 八组约 10–15 GPU 小时。
- P0＋P1 十二组约 16–22 GPU 小时；2 张可用卡建议留 8–12 小时及故障余量，4 张留 4–7 小时及故障余量。
- 单卡也可能赶得上，但没有充足故障余量；实际第一轮日志决定剩余估时。
- 需要 >=32 GiB 空闲磁盘，原模型最佳权重和完整 optimizer/RNG 断点均保存在新 run 目录。不会清理任何旧实验。
- 原 F 峰值 allocated 显存约 12,303 MiB。auto 策略只使用显存占用 <512 MiB、空闲 >=16,000 MiB、利用率 <=10% 的 GPU；单卡单任务，默认最多四个任务。
- 不停止他人任务；没有空闲卡就排队。共享服务器仍应遵守本单位调度/资源规则，不以技术上空闲替代使用权限。

#### 检查、断点与结果

1. 先对每个 G/H 配置用四张真实 TRAIN 图片运行 CUDA identity/gradient smoke，optimizer step 为 0。
2. smoke 检查 CPF 初始化一致、gain0 输出和 task loss 回放、新 head/视觉/骨干梯度，以及 H 的 AE 能反传到新 head。
3. 每轮训练后保存 `last_resume.pth`：完整 model、Adam、Python/NumPy/Torch/CUDA RNG、已完成轮次、best records。`--resume` 仅从完成的 epoch 边界恢复，不覆盖已有完整 run。
4. 若在第一轮保存前中断，没有可恢复断点，会停下并保留证据；不能自动假装已训练。需另建 work 目录重跑该计划。
   若恰在 best 权重已更新、整轮状态尚未提交时中断，best/last 哈希不一致会拒绝自动恢复。完整断点只加载此脚本自己生成的可信文件，不接受外部来源 checkpoint 用作 optimizer/RNG 恢复。
5. 本轮不使用 minority augmentation，因为 native `aug_epoch=37` 大于 10；若修改预算触及 augmentation，会拒绝使用此初版 Full 实现。
6. 配置、源码、资产、split/metadata 校验失败停止，不给绕过 hash 的命令。
7. 每组完成后更新 `.md`、`.csv`、`.json` 和结果 ZIP。每个 checkpoint 固定按 VAL CW AUC 选取，同一权重评价 VAL/TEST CW/OW。TEST 不用于调参。
8. 主表、H−A/B/F、G−E、H−G、sanity 差分、bias0 object/attribute 错误分解由真实结果自动填写。不是因果归因，也不是 best-HM 单一工作点结果。
9. 正结果、负结果都导出；`core_complete` 和 `all_planned_complete` 分开记录。结果容器完整不代表所有任务成功。
10. 默认 ZIP 不含大型 checkpoint，含配置、源码、日志、metrics、逐图 scores、smoke 和权重 SHA256；权重与完整断点须另行保留。

</details>
