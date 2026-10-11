# 18 · residual描述删除扩展与后续候选

状态：仅代码与追加协议；未归档正式结果。

## 修改与结论

保留H_noise10/30/50追加协议和代码，明确描述槽删除范围及与主公式对齐版的不同。

没有对应complete/metrics证据前不标成已训练；其他共享视觉/教师/多seed方案也保持待执行。

## 复现顺序

1. 按 [统一环境与外部输入说明](../../environment/README.md) 准备数据、GloVe、预训练模型。
2. 在仓库根目录校验并生成独立运行目录；此步骤不训练、不覆盖已有目录。

```bash
python tools/reproduce.py verify
python tools/reproduce.py materialize --stage 18 --dest /srv/czsl-reproduce-18
```

3. 运行目录保留历史包内相对位置及哈希。基础工程位于 `project/`，执行包位于运行目录根部。按统一环境说明处理冻结路径和外部输入；路径迁移不能绕过原哈希检查。
4. 优先使用包的 `--root`、`--work` 或环境变量配置路径，先预检/prepare，再执行本阶段训练或诊断。检查结果的 dataset、seed、VAL/TEST、CW/OW、选模与复用身份。
5. [inputs.json](inputs.json) 登记本阶段资产、配置与结果；[runbook.json](runbook.json) 提供去重的实验命令模板及来源；模板路径变量和包内默认路径均须核对。

## 命令变量

以下命令从独立运行目录读取文件。先设置匹配的 Python 与已验证的工程根目录；
历史包内部的默认路径仍须按环境说明处理，不能只替换命令中的变量。

```bash
REPLAY=/srv/czsl-reproduce-18
export CPF_PY="${CZSL_PYTHON:?Set the matching CPF Python executable}"
export PROJECT="${CZSL_PROJECT_ROOT:?Set a verified compatible project root}"
```

## 关键入口

该扩展依赖第 16 阶段的完整 residual 父批次与对应执行包，不依赖第 17 阶段的新训练。
`CZSL_RESIDUAL_WORK` 必须指向已通过父协议校验的完整结果目录。

```bash
WORK=/srv/czsl-runs/residual-noise-new
"$CPF_PY" "$REPLAY/extend_0601_20261011/extension.py" prepare \
  --parent "${CZSL_RESIDUAL_WORK:?Set verified completed residual work}" \
  --release "$REPLAY/reproduce_0601_20261010" --work "$WORK"
"$CPF_PY" "$REPLAY/extend_0601_20261011/extension.py" run \
  --work "$WORK" --export "$WORK.zip" --max-workers 1 \
  --gpus "${CZSL_GPUS:?Set authorized physical GPU IDs}" \
  --deadline "${CZSL_DEADLINE:?Set a future ISO8601 deadline}"
```

执行包：[extend_0601_20261011](<code/extend_0601_20261011>)。

运行时入口：`/srv/czsl-reproduce-18/extend_0601_20261011/`。其原始 README/哈希校验所需说明会从不可变归档标签恢复到运行目录，不在当前仓库重复堆放。

前序阶段：[README.md](<../17-formula-aligned-0601/README.md>)。阶段依赖不等同于每一步都必须重新训练；续训/回放需要对应原权重。

<details>
<summary>模型定义、对照与协议边界</summary>

### 0601：只补三组描述删除实验

本包依赖服务器上已经完成的 `/var/tmp/czsl_0601_20261010`，以及原 r2 发布目录 `/var/tmp/czsl_0601_release_20261010/reproduce_0601_20261010`。校验两者的冻结身份；不重写原批次，不重训 A–H 或四组 sanity。

新任务仅为 `H_noise10`、`H_noise30`、`H_noise50`，各 10 epoch，UT seed 3351，沿用 H 的 ViT/CPF 联合训练、原评价器、VAL CW AUC 选模和同权重 VAL/TEST CW/OW 评价。λ=0.1、β=0.01 均不修改。不会把本次已观察的 TEST 表现用来调参数。

噪声是随机删除 Text/Feature 的 primitive OE/AE 描述，采用原 r2 已实现的固定、嵌套删除序列；原语名字、prior、pair 模板、logic bank 和 OI 不变。因此不能宣称完成全部知识树噪声敏感性。删除比例按每一类描述总槽位数取整，实际索引写入每个 cache 的 manifest。

`prepare` 复制已核验的训练源代码与逻辑 bank 至新的独立目录，记录父批次配置/权重身份/评价文件哈希，产生且只产生三项排定训练。`run` 复用原安全空闲 GPU 队列、检查点恢复和原生评价。不覆盖原结果目录；失败应查看日志并显式 `--resume`。不放宽哈希检查或去掉 smoke。

扩展结果 ZIP 不含模型权重和优化器；这些仍保留服务器，下载包包含权重身份、日志、分数和累计汇总。累计表的前 12 组引用父批次，不是再次训练；3 组全部完成时才输出 `NOISE_EXTENSION_COMPLETE True`。

新复跑必须显式给定未来截止时间；运行中的任务不因截止时间被强制终止。

当前仅有代码与 CPU 检查，不构成三项训练已完成的证据。

</details>
