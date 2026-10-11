# 08 · TRAIN 支持探针、SG-EKR与视觉原型诊断

状态：支持诊断＋3组3轮；VP仅方案。

## 修改与结论

双半折 TRAIN 证据筛支持槽，固定value；另检查排除查询组合的视觉原型。

SG支持描述未优于控制；原型仅开发诊断，不填写为正式VP训练成绩。

## 复现顺序

1. 按 [统一环境与外部输入说明](../../environment/README.md) 准备数据、GloVe、预训练模型。
2. 在仓库根目录校验并生成独立运行目录；此步骤不训练、不覆盖已有目录。

```bash
python tools/reproduce.py verify
python tools/reproduce.py materialize --stage 08 --dest /srv/czsl-reproduce-08
```

3. 运行目录保留历史包内相对位置及哈希。基础工程位于 `project/`，执行包位于运行目录根部。按统一环境说明处理冻结路径和外部输入；路径迁移不能绕过原哈希检查。
4. 优先使用包的 `--root`、`--work` 或环境变量配置路径，先预检/prepare，再执行本阶段训练或诊断。检查结果的 dataset、seed、VAL/TEST、CW/OW、选模与复用身份。
5. [inputs.json](inputs.json) 登记本阶段资产、配置与结果；[runbook.json](runbook.json) 提供去重的实验命令模板及来源；模板路径变量和包内默认路径均须核对。

## 命令变量

以下命令从独立运行目录读取文件。先设置匹配的 Python 与已验证的工程根目录；
历史包内部的默认路径仍须按环境说明处理，不能只替换命令中的变量。

```bash
REPLAY=/srv/czsl-reproduce-08
export CPF_PY="${CZSL_PYTHON:?Set the matching CPF Python executable}"
export PROJECT="${CZSL_PROJECT_ROOT:?Set a verified compatible project root}"
```

## 关键入口

先核对 TRAIN 支持掩码与前序权重，再执行匹配的三组 SG-EKR；其他探针见 runbook.json：

```bash
"$CPF_PY" "$REPLAY/sgekr_support_mask_20261008/verify_release.py"
bash "$REPLAY/sgekr_support_mask_20261008/run.sh" all
```

执行包：[sgekr_support_mask_20261008](<code/sgekr_support_mask_20261008>)。

运行时入口：`/srv/czsl-reproduce-08/sgekr_support_mask_20261008/`。其原始 README/哈希校验所需说明会从不可变归档标签恢复到运行目录，不在当前仓库重复堆放。

前序阶段：[README.md](<../07-ekr-v2/README.md>)。阶段依赖不等同于每一步都必须重新训练；续训/回放需要对应原权重。

原始结果：[results.csv](<results/artifacts/reports/results.csv>)。

配置文件：[config_eval_effective.yml](<configs/artifacts/baseline/val_cw/config_eval_effective.yml>)、[config_eval_effective.yml](<configs/artifacts/baseline/val_ow/config_eval_effective.yml>)、[config_eval_effective.yml](<configs/artifacts/runs/G_name/epochs/epoch_01_cw/attempt_0000/config_eval_effective.yml>)、[config_eval_effective.yml](<configs/artifacts/runs/G_name/epochs/epoch_02_cw/attempt_0000/config_eval_effective.yml>)；已生成的运行配置和 protocol 也在结果中，完整文件见 inputs.json。

<details>
<summary>模型定义、对照与协议边界</summary>

### SG-EKR seed-0 pilot

This release runs a bounded support-gated fixed-value EKR pilot on the reviewed UT-Zappos
artifacts. It preserves the old name pilot, failed v2 result, source tree, banks, checkpoints,
and gates. It does not use the random semantic graph route.

The prepare phase validates every pinned input, reconstructs a deterministic 3/3 image split
inside each six-image TRAIN pair, computes object-conditioned support for every AE slot, and
freezes slots whose support is at least 0.005 on both halves. VAL and TEST are not used to
construct the mask.

The three matched variants are `G_name`, `G_support`, and `G_shuffle`. All use fixed raw values
and train only 28,834 query/key/null/gain parameters for three epochs. Epoch zero is always a
selection candidate. The single-seed operational threshold is +0.10 CW AUC percentage points
over the common name anchor, the matched name router, and the matched shuffle router.

Run phases separately after extraction and release verification:

```bash
bash ${REPLAY}/sgekr_support_mask_20261008/run.sh prepare
bash ${REPLAY}/sgekr_support_mask_20261008/run.sh train
bash ${REPLAY}/sgekr_support_mask_20261008/run.sh report
```

If training is interrupted after at least one committed epoch:

```bash
bash ${REPLAY}/sgekr_support_mask_20261008/run.sh resume
bash ${REPLAY}/sgekr_support_mask_20261008/run.sh report
```

Read the outcome:

```bash
cat /var/tmp/czsl_xhw_sgekr_20261008/reports/decision.json
cat /var/tmp/czsl_xhw_sgekr_20261008/reports/results.csv
cat /var/tmp/czsl_xhw_sgekr_20261008/support_mask.json
```

The report phase creates `/var/tmp/czsl_xhw_sgekr_20261008_review.zip`. The archive excludes
weights; retain all server checkpoints. Do not run the old stage C even if this pilot passes.
A passing result only permits a separately frozen router-seed confirmation protocol.

</details>
