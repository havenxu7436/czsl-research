# 06 · EKR 机制诊断

状态：既有权重回放；0新训练。

## 修改与结论

检查 value、null、对象上下文、置换与margin；保护名称基底并确认退化原因。

弱对象依赖和null拒识问题促成共享value对齐；诊断不能直接证明知识泛化。

## 复现顺序

1. 按 [统一环境与外部输入说明](../../environment/README.md) 准备数据、GloVe、预训练模型。
2. 在仓库根目录校验并生成独立运行目录；此步骤不训练、不覆盖已有目录。

```bash
python tools/reproduce.py verify
python tools/reproduce.py materialize --stage 06 --dest /srv/czsl-reproduce-06
```

3. 运行目录保留历史包内相对位置及哈希。基础工程位于 `project/`，执行包位于运行目录根部。按统一环境说明处理冻结路径和外部输入；路径迁移不能绕过原哈希检查。
4. 优先使用包的 `--root`、`--work` 或环境变量配置路径，先预检/prepare，再执行本阶段训练或诊断。检查结果的 dataset、seed、VAL/TEST、CW/OW、选模与复用身份。
5. [inputs.json](inputs.json) 登记本阶段资产、配置与结果；[runbook.json](runbook.json) 提供去重的实验命令模板及来源；模板路径变量和包内默认路径均须核对。

## 命令变量

以下命令从独立运行目录读取文件。先设置匹配的 Python 与已验证的工程根目录；
历史包内部的默认路径仍须按环境说明处理，不能只替换命令中的变量。

```bash
REPLAY=/srv/czsl-reproduce-06
export CPF_PY="${CZSL_PYTHON:?Set the matching CPF Python executable}"
export PROJECT="${CZSL_PROJECT_ROOT:?Set a verified compatible project root}"
```

## 关键入口

在原路径兼容和前序 checkpoint 核对完成后，执行冻结诊断，不产生新训练：

```bash
"$CPF_PY" "$REPLAY/name_ekr_mechanism_diag_20261007_v1/verify_release.py"
bash "$REPLAY/name_ekr_mechanism_diag_20261007_v1/run.sh"
```

执行包：[name_ekr_mechanism_diag_20261007_v1](<code/name_ekr_mechanism_diag_20261007_v1>)。

运行时入口：`/srv/czsl-reproduce-06/name_ekr_mechanism_diag_20261007_v1/`。其原始 README/哈希校验所需说明会从不可变归档标签恢复到运行目录，不在当前仓库重复堆放。

前序阶段：[README.md](<../05-name-anchor-ekr/README.md>)。阶段依赖不等同于每一步都必须重新训练；续训/回放需要对应原权重。

## 证据入口

以下链接指向归档原文件；训练、回放和诊断范围以上述阶段状态为准。

- [results/artifacts/integrity_checks.json](<results/artifacts/integrity_checks.json>)
- [results/artifacts/mechanism_findings.json](<results/artifacts/mechanism_findings.json>)

<details>
<summary>模型定义、对照与协议边界</summary>

### 失败 pilot 后的冻结机制诊断

日期：2026-10-07。目标：解释当前实现为何缺乏正确描述优势，决定是否值得另立改进版本。当前阶段 B `proceed=false` 保持原样，不运行原阶段 C/D。

#### 这一轮具体做什么

复用 seed 0 的 N_name、N_clean、N_shuffle 三项最佳分支和同一个冻结 A_unit。所有参数保持冻结；没有 optimizer、反向传播或新增训练。

缓存冻结 CPF 在以下三个数据视图上的基础分数和属性特征，再从缓存计算分支：

| 数据视图 | 图像 | 用途 |
|---|---|---|
| TRAIN 平衡子集 | 每个 TRAIN pair 固定随机取至多 6 张，83 pair，总数至多 498 | 查看分支在训练标签上的分数差与 CE；采用确定性的 VAL 图像变换 |
| VAL CW | 全部 3214 张 | 核对旧指标和分支作用，使用原生 CW 候选掩码 |
| VAL OW | 相同 3214 张 | 检查 seen/unseen 取舍和对象条件作用 |

TRAIN 选择 seed 固定为 20261007，不按模型错误挑图。这个子集使用固定视图、按 pair 平衡，不能当作全部 TRAIN 的总体指标，也不能直接与训练时随机增强的 loss 数值比较。

每个视图有 9 项纯推理诊断，共 27 项缓存分支计算；仅两项 VAL 视图计算原生 AUC/HM，即 18 次指标计算。图像主干只在三个视图上各跑一次。

| 项目 | 数量 | 检查的问题 |
|---|---:|---|
| 三项原始分支 | 3 | 重新加载实际权重是否复现阶段 B |
| N_clean 描述错配，固定 seed 11/22/33 | 3 | 所有属性的 AE 槽更换成其他属性的描述，保留自身名称；空槽、分数和 margin 是否响应 |
| N_name 描述错配，seed 11 | 1 | 名称对照不应受描述更换影响，否则停止 |
| N_clean 对象上下文错配，seed 11 | 1 | 只重排分支的对象名称上下文；基础分数和候选身份不变，检验条件查询的实际影响 |
| N_clean 去空槽并重归一化 | 1 | 仅在推理时移除已有空槽，查看它当前实际贡献 |

错配是无固定点的确定性排列，会记录 receiver→donor 映射。三组输入排列不是三个训练 seed；所有结果都报告，不按效果选排列。这些对照没有构建随机语义图，仍属于原 EKR 路线的机制诊断。推理时去空槽也不是完成了原阶段 D 的重新训练消融。

程序会打印原 dataset 的 TRAIN/VAL/test 数量。这只是沿用原数据集类读取 split 元数据；本脚本只加载 TRAIN 子集和 VAL 图像，不加载 test 图像、不计算 test 指标。

#### 完整性与诊断验收

```bash
cat /var/tmp/czsl_xhw_name_ekr_diag_20261007_v1/integrity_checks.json
cat /var/tmp/czsl_xhw_name_ekr_diag_20261007_v1/mechanism_findings.json
```

完整性文件中应看到 `passed=true`、`new_training=false`、`base_state_unchanged=true`、`name_input_control_passed=true`。原始三项分支的原生指标与阶段 B 的最大允许回放差异为 0.05 pp；这是复现容差，不是收益门槛。

另看 `raw_argmax_exact_replay`。若为 false，意味着存在原始 top1 的细小回放差异，应先检查 `replay_checks.json`，不能不加说明地当作旧逐图结果完全一致。

程序不会生成一个新的研究 `proceed=true`。诊断通过仅表示计算与来源核对通过，旧 pilot 仍为 false。

#### 诊断之后如何决定

先看全部数据，而不是选单个最好指标。主要问题是：

| 观察 | 解释边界 | 后续方向 |
|---|---|---|
| N_clean 在 TRAIN 固定视图的 margin/CE 也没有优于名称对照 | 当前描述值的判别作用不足；不能据此断言文本内容本身无效 | 优先查描述值与冻结属性特征的对齐，再设计可学习值对齐或属性模块适配的新版本 |
| TRAIN 有作用，VAL 很弱或相反 | 更像训练样本适配未转化成泛化收益，尚不是严格的过拟合证明 | 检查 seen/unseen 分组与损失设计，避免直接扩大 rank、轮数或数据集规模 |
| 描述全部错配后空槽也不响应 | 当前平均取舍机制缺少可靠性响应 | 下一版检验仅用 TRAIN 构造的知识不匹配监督；响应本身也不等于置信度已校准 |
| 对象上下文错配几乎不改变分支分数 | 对象条件的作用很弱 | 查查询是否忽略对象输入；不能把已添加对象输入直接写成已验证的条件性创新 |
| 去空槽与原结果几乎一致 | 当前空槽贡献有限 | 不直接展开原阶段 D 的去空槽重新训练；先解决取舍信号的问题 |

margin 定义采用 `正确组合分数 - 最强错误组合分数`。脚本同时记录固定原竞争者下的残差差值与新竞争者变化后的实际 margin，避免把竞争者切换误判为收益。标签仅用于离线统计，不进入查询、知识选择或部署预测。

这一轮不会临时调阈值、选择排列或挑权重。若诊断支持新的机制假设，再单独建立 EKR v2 的匹配对照和有限预算；原阶段 B 的负结果继续保留。当前不承诺改进版一定成功，也不自动转入随机语义图路线。

</details>
