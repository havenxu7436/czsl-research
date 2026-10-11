# 10 · CPF＋LogiCZSL B/F直接结合

状态：已训练：UT3351、四组10轮。

## 修改与结论

补A_ref/B/E_ref/F，确保逻辑损失实际加入反传，冻结β=0.01、q=3。

当前直接结合未产生稳定配对正增量；逻辑方向以实际服务器源码为准。

## 复现顺序

1. 按 [统一环境与外部输入说明](../../environment/README.md) 准备数据、GloVe、预训练模型。
2. 在仓库根目录校验并生成独立运行目录；此步骤不训练、不覆盖已有目录。

```bash
python tools/reproduce.py verify
python tools/reproduce.py materialize --stage 10 --dest /srv/czsl-reproduce-10
```

3. 运行目录保留历史包内相对位置及哈希。基础工程位于 `project/`，执行包位于运行目录根部。按统一环境说明处理冻结路径和外部输入；路径迁移不能绕过原哈希检查。
4. 优先使用包的 `--root`、`--work` 或环境变量配置路径，先预检/prepare，再执行本阶段训练或诊断。检查结果的 dataset、seed、VAL/TEST、CW/OW、选模与复用身份。
5. [inputs.json](inputs.json) 登记本阶段资产、配置与结果；[runbook.json](runbook.json) 提供去重的实验命令模板及来源；模板路径变量和包内默认路径均须核对。

## 命令变量

以下命令从独立运行目录读取文件。先设置匹配的 Python 与已验证的工程根目录；
历史包内部的默认路径仍须按环境说明处理，不能只替换命令中的变量。

```bash
REPLAY=/srv/czsl-reproduce-10
export CPF_PY="${CZSL_PYTHON:?Set the matching CPF Python executable}"
export PROJECT="${CZSL_PROJECT_ROOT:?Set a verified compatible project root}"
```

## 关键入口

执行包：[logiczsl_cpf_bf_20261008](<code/logiczsl_cpf_bf_20261008>)。

运行时入口：`/srv/czsl-reproduce-10/logiczsl_cpf_bf_20261008/`。其原始 README/哈希校验所需说明会从不可变归档标签恢复到运行目录，不在当前仓库重复堆放。

前序阶段：[README.md](<../09-budget10/README.md>)。阶段依赖不等同于每一步都必须重新训练；续训/回放需要对应原权重。

知识版本：[knowledge.json](<../../knowledge/ut-zap50k/logic/reconstructed-v1/knowledge.json>)。

<details>
<summary>模型定义、对照与协议边界</summary>

### CPF + LOGICZSL B/F 独立执行包

日期：2026-10-08。范围：UT-Zappos、原 CPF/GloVe、A_ref/B/E_ref/F。不覆盖旧 ACDE，不修改 EKR。

这是依据原论文公式重建逻辑损失、以新知识资产适配 CPF 的实验包，不是原始 Logic-Troika 结果的精确复现。执行参数见 runbook.json 与脚本入口。

#### 包含内容

- `assets/ut-zap50k/knowledge.json`：12 对象、16 属性、5 个 OI 父类、每个 primitive 三条 OE/AE 描述。按论文方法重建，需要进一步审核，不是原作者遗失数据。
- `model/logic_loss.py`：论文 Eq.6/9/12，q=3；与作者 `logicloss.py` 快照数值对照通过。
- `model/bf_logic.py`：复用 CPF 视觉分支与 cosine classifier；固定 GloVe 知识节点；无新增可训练参数；β=0.01 只乘一次。
- `prepare.py`：校验原源码版本，只在独立目录复制并接入逻辑，生成四组配置与冻结协议。
- `smoke.py`：被复制到新源码的 `tests/smoke_bf.py`，真实 TRAIN 图像 CUDA 恒等/梯度检查。
- `run.py`：训练 10 轮、按 VAL CW AUC 选模、评价 VAL CW/OW；不提供 TEST 调参入口。
- `summarize.py`：将真实训练证据和配对结果写成 `.md`、`.csv`，不把未完成任务写成成功。
- `test_release.py`：CPU 公式、梯度、源码补丁前向、准备/汇总流程测试。随机张量与流程 fixture 不是训练结果。

#### 复现边界

保留原论文的三项逻辑损失、三条描述、q=3、β=0.01；适配层改为现有 CPF 的 GloVe 和视觉分支，保留 CPF 的 `L_comp + 0.4 L_obj + 0.6 L_attr` 与 legacy 推理方式。训练预算沿用 ACDE 的 10 轮，而不是论文 UT 的 15 轮；F 保留旧 E 的文本缓存与 λ=0.1。新知识只进入逻辑损失，不用于替换旧 E 的文本/校准缓存。

所有需要覆盖已有配置、bank、训练结果的操作都会拒绝执行。源码版本守卫失败时，不要删除守卫或覆盖原源码，请保留报错核对版本。

#### 来源

- [LOGICZSL，CVPR 2025 原论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_LOGICZSL_Exploring_Logic-induced_Representation_for_Compositional_Zero-shot_Learning_CVPR_2025_paper.pdf)，§3.2-3.4、Eq.6/9/12/13、Fig.4。
- [作者仓库](https://github.com/Pieux0/LOGICZSL)，核查日期 2026-10-08。
- 归档的 `LogiCZSL_copy.tar.gz` 中 `model/logicloss.py` 用于公式数值核对；没有在包中声称恢复原作者知识 JSON。

</details>
