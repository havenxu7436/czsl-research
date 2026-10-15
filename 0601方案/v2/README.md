# v2 · 0601 统一修订版

> **[复现公式与代码v2：完整公式与核心代码](复现公式与代码v2.md)** · [v1修订意见：逐项审计与修订依据](v1修订意见.md)
>
> 全部已识别公式问题在同一版本统一修订。v1、r2 与已有结果保持不变；v2 是修订后的新模型，不是原 LogiCZSL 数据/成绩的完整复现。

状态：seed3351的32组矩阵已完成，每组10 epoch，共128份评价，使用20261014修正版。seed3352、3353未纳入本次归档。指标表见下方[结果与文件](#结果与文件)。本页只整理数据与入口，不进行结果分析。

原20261013发布包及34项CPU验证记录保留不变；修正版原包内仍名为 `revised_0601_20261013`，为避免与旧版覆盖，仓库分区和materialize运行目录单独命名为 `revised_0601_20261014_fix`。修正版随包CPU记录为40项通过，不冒充此次导入重新执行CPU/CUDA测试。

## 实验设置

- UT-Zappos；seeds=3351/3352/3353；ViT B16、GloVe300；batch64；每组统一10epochs。
- 联合概率：对象 `log_softmax` + 属性条件 `log_softmax(U + lambda*prior)`；训练和推理一致。
- 视觉知识 head：rank128 图像—对象—属性交互 + 动态 pair 相似度；保留 native CPF 条件基底。
- 语义约束：完整 joint 边缘概率、互斥 child 概率和、JS 一致性及 TRAIN primitive 标签锚定；不是原 fuzzy 算子的原样复现。
- 所有比较组 fresh 训练，首次严格最大 VAL CW AUC 选模，同一 checkpoint 评价 VAL/TEST CW/OW。不混入旧30/10/10epoch结果。

## 流程

1. `materialize` 恢复分区存储的完整包及必要历史工程输入，不会开始训练。
2. 运行 CPU 检查，冻结新 work 下的代码、配置、知识、数据 split 与 hash。
3. 每个修订模型先进行4张真实 TRAIN 图像的 CUDA smoke；通过才正式训练。
4. 队列完成后，VAL 固定 bias，分析 TEST primitive 错误与同权重 prior 读出；再次导出汇总。本批快照尚未包含这些后分析文件。

上述是同一完整修正版的验证流程，不是分步修复公式。失败保留证据，不删除旧目录，也不因效果不涨停止。

## 运行入口

变量与路径约定沿用 [环境说明](../../environment/README.md)。在仓库根目录执行，先准备好 `PROJECT` 的原 CPF 源码、UT图像/split、GloVe、ViT权重和原 cache。冻结旧资产中的绝对路径仍需匹配，`materialize` 不自动迁移全部外部输入。

```bash
REPLAY=/srv/czsl-reproduce-v2
python tools/reproduce.py materialize --line 0601方案 --stage v2 --dest "$REPLAY"
V2_PKG="$REPLAY/revised_0601_20261014_fix"
"$CPF_PY" "$V2_PKG/test_revision.py"
"$CPF_PY" "$V2_PKG/prepare_revision.py" --root "$PROJECT" --work "$WORK" \
  --suite all --epochs 10 --seeds 3351,3352,3353
"$CPF_PY" "$V2_PKG/run_suite.py" --work "$WORK" \
  --gpus "${CZSL_GPUS:?Set physical GPU IDs}" --max-workers 3 \
  --export "$WORK/results.zip"
```

以上为前台队列；需脱离 SSH 持续运行时，将最后一条命令在 tmux 中运行。`--gpus` 是候选物理卡，脚本仍检查空闲状态，不终止其他任务。确认无遗留 worker/process group 后才可用 `--resume`；没有完整 last_resume checkpoint 时不得假装无损续训。不可修改冻结代码/配置后继续旧 work。

后分析与再次导出：

```bash
"$CPF_PY" "$V2_PKG/posthoc_revision.py" --work "$WORK"
"$CPF_PY" "$V2_PKG/run_suite.py" --work "$WORK" --summarize-only \
  --export "$WORK/results.zip"
```

选模仅用 VAL；posthoc bias 在 VAL 上选，TEST只应用。结果ZIP不含模型checkpoint/图像大文件，checkpoint留在服务器。

## 配置矩阵

| suite | variants | seeds | 任务数 | 每组epochs | 初始磁盘预算 |
|---|---:|---:|---:|---:|---:|
| core | 13 | 3 | 39 | 10 | 88GiB |
| small（独立运行，含 fresh Rfull） | 20 | 3 | 60 | 10 | 130GiB |
| all（Rfull只跑一次） | 32 | 3 | 96 | 10 | 202GiB |
| external（须实际 LLM 资产） | 3 | 3 | 9 | 10 | 28GiB |

主矩阵包含 A_native、R0、Rtext、Rprior、Rmvp、Rexternal、Rmvp_logic、Rfeature、Rfull及4个sanity对照。小实验覆盖1/3/5描述、1/2/3树深、10/30/50%描述删除、聚合方式、prior变换、可学习lambda、延迟beta、属性边缘CE、动态/静态head、温度与模块删除。

配置目录保存 **34种 variant 的预览 YAML**（all32 + external专属2）。正式 seed/work 配置由 `prepare_revision.py` 生成，不能绕过冻结流程直接使用预览。代码与配置按旧仓库格式分区，恢复执行包后才运行。

初始排期曾按旧6.2–6.5min/epoch估计all约38–48小时、core约15–19小时；这不是当前共享服务器的完成时间承诺，实际逐轮耗时见本批train.log。磁盘按10+2×剩余任务数GiB预留，恢复时也检查，不绕过质量/空间护栏。

## 结果与文件

| seed | 每组实际 epoch | 训练组数 | 评价份数 | 结果入口 |
|---|---:|---:|---:|---|
| 3351 | 10 | 32 | 128 | [指标与训练记录](results/seed3351-10epoch-20261015/README.md) |
| 3352 | — | — | — | 未纳入本次归档 |
| 3353 | — | — | — | 未纳入本次归档 |

seed3351按首次严格最大VAL CW AUC选模，同一权重评价VAL/TEST CW/OW。表格单位、列顺序和结果页结构沿用02公式对齐版；未导出的seed不填0，不生成跨seed统计。原始记录保持不变，未执行posthoc分析。

- [本批实际修正版执行代码](code/revised_0601_20261014_fix/)
- [修正版所有 variant 预览配置](configs/revised_0601_20261014_fix/configs/)
- [修正版配置开关索引](configs/revised_0601_20261014_fix/configs/variant_options.json)
- [原始冻结work配置（含96组计划，不等于96组完成）](configs/seed3351-10epoch-20261015/configs/)
- [seed3351：指标与训练记录](results/seed3351-10epoch-20261015/README.md)
- [seed3351：128份完整评价指标](results/seed3351-10epoch-20261015/完整评价指标.md)
- [seed3351：原实验设计命名对照](results/seed3351-10epoch-20261015/实验命名对照.md)
- [seed3351：归档与校验说明](results/seed3351-10epoch-20261015/归档与校验.md)
- [知识、配置与来源索引](inputs.json)
- [具体执行命令](runbook.json)
- [本地CPU检查与独立解压验证](results/README.md)
- [原部署ZIP（Git LFS）](results/deployment/0601_unified_revision_20261013.zip)
- [部署ZIP SHA256](results/deployment/0601_unified_revision_20261013.zip.sha256)
- [本批实际使用的20261014修正版部署ZIP（Git LFS）](results/deployment/0601_unified_revision_20261014_fix.zip)
- [修正版部署ZIP SHA256](results/deployment/0601_unified_revision_20261014_fix.zip.sha256)

原包README和旧引用资料以原字节保存于 package support；仓库说明只使用相对链接与可配置路径。`materialize --stage v2` 分别恢复旧包、修正版和本批work快照，修正版125项manifest及导出1765项manifest可独立校验。原发布记录与公式文档不覆盖，本批[导入记录](../../provenance/updates/20261015_v2_seed3351.json)另存。恢复的work含原服务器绝对路径且不含checkpoint，不应当作可直接无损续训的备份。

## 未完成范围

LLM-only/mixed 有严格资产接口，但需独立全组合评分和provenance；空模板/seen-unseen标签不算LLM结果。MIT/C-GQA、CLIP双编码器语义对齐和跨benchmark矩阵仍未接入此包。定性失败原因需要人工看图，不由统计错误类型自动生成。
