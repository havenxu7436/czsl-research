# LOGICZSL 知识与逻辑损失重建，以及 CPF 的 B/F 训练操作手册

交付日期：2026-10-08。范围已按你的选择收敛到 **先完成 CPF 的 B/F**，不同时重跑独立 Troika/CLIP。

## 1. 本次实际完成与尚未完成

已经完成：核对原论文关键页与公式、审计本地 LogiCZSL 源码、重建 UT 知识 JSON、实现三项逻辑损失与 CPF 接口、生成独立准备/训练/汇总脚本、本地 CPU 测试、用真实 GloVe 验证知识编码。

**尚未完成：真实 TRAIN 图像 CUDA 冒烟，以及 A_ref/B/E_ref/F 的完整 GPU 训练。** 当前机器没有可用 CUDA；无交互只读 SSH 尝试返回 `Permission denied (publickey,password,keyboard-interactive)`。因此本文是已测试的实现与执行手册，不是已经得到 B/F 效果的实验报告。

训练跑完后，`summarize.py` 会自动生成另一份包含实际指标的 Markdown。不要把本文的公式检查通过当成 B/F 有收益。

## 2. 能复现什么，不能声称什么

原作者遗失的历史 LLM 输出、历史采样设置和原始知识文件，不能从论文反推为逐字一致的数据。这里复现的是 **论文的知识结构和逻辑损失方法，并适配现有 CPF**。

| 部分 | 本次处理 | 与原论文的一致性 |
|---|---|---|
| OI 对象归纳树 | 新知识 JSON：对象到一个父类，深度 2 | 结构遵循论文；父类分组不是历史原数据 |
| OE 对象扩展树 | 每个对象三条区分性特征短语 | 方法遵循论文；文字为助手重建 |
| AE 属性扩展树 | 每个属性三条语义/材质描述 | 方法遵循论文；文字为助手重建 |
| 逻辑损失 | Eq.6/9/12，q=3，β=0.01 | 与本地作者公式数值对照通过 |
| 知识节点编码 | 固定 GloVe 6B 300d 短语均值并归一化 | 是 CPF 适配，不是原论文 CLIP 编码 |
| 图像分支 | 现有 CPF 的对象/属性视觉表示 | 不换 backbone；不声称等同 Logic-Troika |
| 分类损失 | 保留 CPF 原有权重：对象 0.4、属性 0.6 | 不替换为论文 α1=α2=0.1，以隔离新增逻辑效果 |
| 推理与评估 | 保留原 CPF legacy 分数与现有评价器 | 不同时替换成论文 Eq.15/16 |
| 训练预算 | 四组各 10 轮、batch 64，沿用 ACDE 优化参数 | 为公平比较 CPF，非论文 UT 的 15 轮 |
| 数据集 | 复用原 UT 图像、元数据和标准划分 | 不重建图像，不换划分 |

可用的报告称谓：**“CPF + LOGICZSL 三项逻辑损失，基于论文方法重建知识资产”**。不应写“精确复现原作者数据”或“复现了论文 45.8% AUC”。论文数值属于 CLIP/Troika 的 TEST，当前是 CPF/GloVe 的 VAL，不能直接比较。

## 3. 本地作者源码为什么不能直接拿来训练

核对文件：用户的 `C:/Users/llenovo/Desktop/CZSL/CZSL_code/LogiCZSL_copy.tar.gz`。

1. `model_factory.py` 选择 `model.troika.Troika`，其中三项逻辑分支和逻辑损失被注释掉。
2. `logic_copy.py` 虽然计算 `loss = loss_ce + loss_logic`，但 `loss_calu()` 返回的是 `[loss_ce, ...]`；训练入口对返回的第一项反传，不能据此认为三项逻辑已参与训练。
3. `dataset.py` 使用 `oi_parent/oe_child/ae_child`，训练入口仍读取 `hc_parent/oda_child/ada_child`；存在命名不一致。
4. OI/OE 文件路径包含 `<your_oi_dict.json>`、`<your_oe_dict.json>` 占位符；原知识 JSON 不在这个压缩包中。
5. `logic_copy.py` 还有辅助知识节点 CE 和按 batch 编号的开关，而论文主公式是 `L_ce + beta*(L_oi+L_oe+L_ae)`。本轮以主论文公式为准，不盲目继承副本里的调试行为。

因此，本包使用作者 `logicloss.py` 核对损失公式，但没有用“删除开关限制”冒充完成接入。

## 4. 知识资产的内容、审核与冻结

交付文件 `assets/ut-zap50k/knowledge.json` 覆盖 12 个对象、16 个属性；5 个父类；36 条 OE 和 48 条 AE 描述。没有使用验证/测试图像、验证/测试指标生成描述。

分组例子：三种 Boots 属于 `boots`；Boat Shoes/Flats/Loafers/Oxfords/Sneakers 属于 `closed low shoes`；Clogs and Mules/Slippers 属于 `slip on comfort footwear`。Sandals、Heels 分别有单独父类。这是可审核的语义抽象，不声称恢复了作者原分组；单成员父类带来的约束效果也是一个限制。

OE 例如：Ankle Boots 描述短靴筒、踝部高度；Knee High Boots 描述膝部高度；Heels 描述后跟抬高。AE 例如：Patent Leather 对应光泽涂层、反光表面；Suede 对应刷毛/绒面；Faux Leather 与 Leather 的部分线索重叠，不能保证通过照片区分真实材料。

审核建议：先读 JSON，检查父类是否合理、三条描述是否重复或含不可靠断言。**包内状态仍为 `assistant_reconstructed_requires_user_review`。** `--allow-reconstructed` 只是明确接受使用重建知识开展探索，不会把它变成官方数据或自动标成“人工全量审核通过”。

若修改描述，务必在准备实验之前修改，并重新生成发布包和校验和；已有 bank/配置一旦冻结，不要原地改。改动后用新的版本目录，本轮与新轮不要混用。

本地真实 `glove.6B.zip` 编码检查：这些父类与描述词覆盖率均为 100%。这只说明词能编码，不说明短语均值保留了足够语义，也不证明知识有效。

原始论文 Fig.4 的重建思路：将对象归纳成有共同类别名的组；为每个对象生成相对其余类别有区分力的三条视觉特征；为每个属性生成三个相关替换词/短语。这里没有调用付费 LLM API，也没有冒称使用历史 GPT-4。若后续改用 LLM，记录服务方、实际模型 ID、调用日期、完整 prompt、原始响应与每次人工改动，再冻结 JSON。

## 5. 逻辑损失如何进入 CPF

定义每条蕴含的违反度 `v = s_antecedent*(1-s_consequent)`，批次上的聚合为：

$$
M_q(v)=\left(\frac1B\sum_{b=1}^B v_b^q\right)^{1/q},\qquad q=3.
$$

- OI 向上：对象应该支持它的父类，对每个对象计算 `M_q(s_obj*(1-s_parent))` 再平均。
- OI 向下：父类应该支持至少一个子对象，计算 `M_q(s_parent*(1-max(s_children)))` 再平均。
- OE：每个对象对应的三个描述分别支持该对象，平均三条蕴含违反度。
- AE：每个属性对应的三个描述分别支持该属性，平均三条蕴含违反度。

$$
L_{total}=L_{CPF}+0.01\,(L_{OI}+L_{OE}+L_{AE}).
$$

对象/属性 logits 使用 CPF 原生分类结果；父类和 OE logits 使用原生对象视觉表示，AE logits 使用原生属性视觉表示。知识节点向量固定，但图像表示和原有模型按原协议训练。

重要的 softmax 轴：OE/AE 张量分别是 `[B,O,3]`、`[B,A,3]`，每个描述槽在对象/属性类别轴上 softmax；**不是三个描述槽之间 softmax**。这与本地作者 `logicloss.py` 一致。

没有新增可训练 head，没有额外知识节点 CE，没有新 backbone，没有随机图，没有修改推理分数。β 只乘一次。`beta=0` 跳过知识前向；数值实现用向量范数计算广义均值，避免零违反度时的幂根梯度 NaN。

## 6. 本轮实验组和公平比较

| 组 | 文本 | 旧校准 | 三项逻辑 | 主要比较 |
|---|---|---|---|---|
| A_ref | 关 | 关，λ=0 | 关 | 同协议 CPF 参考 |
| B | 关 | 关，λ=0 | 开，β=0.01 | B − A_ref |
| E_ref | 开 | 开，λ=0.1 | 关 | 同协议旧 E 参考 |
| F | 开 | 开，λ=0.1 | 开，β=0.01 | F − E_ref，以及 F − A_ref |

**F 不等于 C+Logic，也不是 λ=0 的文本+逻辑版本。** 本包没有偷偷把 F 改名。F 的文本和校准使用旧 `cache_trusted_v1/ut-zap50k`；新重建知识仅进入逻辑项，B/F 共用同一份新逻辑 bank。两条知识路径的版本分别记录，不能写成“全部知识来自同一份新 JSON”。

为何补 A_ref/E_ref：原训练入口没有显式设置 Python 主进程随机种子，且后来打开 cudnn benchmark。本包统一设置 Python/NumPy/Torch 种子、关闭 benchmark；所以需要同协议参考排除这些工程变化，旧 A/E 结果仍保留为历史记录。GPU 跨设备/驱动仍不保证逐 bit 相同。

第一轮 seed=3351，共 4 次×10 轮。从初始化开始，不接着已有 A/E checkpoint 多训。初轮完成后再固定知识/β/q补 seed=0、2026；不要看到某一 seed 有收益才挑选报告。

## 7. Windows 上传

在 **Windows PowerShell** 执行。地址沿用已有实验记录；若实际 SSH 地址或端口不同，使用你的实际连接配置。密码在 SSH 提示中输入，不写进脚本，也不要发给助手。

```powershell
$BFPackage = 'C:\Users\llenovo\Documents\ChatGPT\CZSL\output\logiczsl_cpf_bf_20261008.zip'
$BFChecksum = "$BFPackage.sha256"
Get-FileHash -Algorithm SHA256 -LiteralPath $BFPackage
Get-Content -LiteralPath $BFChecksum
scp $BFPackage $BFChecksum xhw@211.87.232.122:/tmp/
```

## 8. Linux 解压、版本校验与准备

在 **Linux 服务器终端**执行。沿用现有 CPF Python，不重新安装一套覆盖环境。

```bash
(
  set -euo pipefail
  cd /tmp
  sha256sum -c logiczsl_cpf_bf_20261008.zip.sha256
  test ! -e /tmp/logiczsl_cpf_bf_20261008
  /data/xhw/miniconda3/envs/cpf/bin/python -m zipfile -e \
    /tmp/logiczsl_cpf_bf_20261008.zip /tmp
  /data/xhw/miniconda3/envs/cpf/bin/python \
    /tmp/logiczsl_cpf_bf_20261008/verify_release.py
)
```

只在第一次解压时执行以上整块；已经正确解压后只执行 `verify_release.py`。重复解压会被存在性检查拦住。

先检查环境、路径和磁盘。下列路径是已有记录，不代表我已经成功登录服务器验证：

```bash
(
  set -euo pipefail
  BF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
  test -x "$BF_PY"
  test -s /data/xhw/czsl_repro/src/KCC0601_v1/models/kcc_cpf.py
  test -s /data/xhw/czsl_repro/data/glove/glove.6B.300d.txt
  test -s /data/xhw/czsl_repro/knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k/labels.json
  "$BF_PY" -c 'import torch,yaml,numpy; print(torch.__version__); print("CUDA",torch.cuda.is_available()); assert torch.cuda.is_available()'
  nvidia-smi
  df -h /var/tmp
)
```

先为四组预留至少约 8 GiB 工作空间并检查 GPU 空闲；这是保守起点，不是磁盘占用保证。脚本每次训练还会检查剩余空间，不会自动清理任何旧结果。

**读完并接受重建知识的探索性质后**，准备 seed 3351：

```bash
(
  set -euo pipefail
  BF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
  BF_BUNDLE=/tmp/logiczsl_cpf_bf_20261008
  "$BF_PY" "$BF_BUNDLE/prepare.py" \
    --root /data/xhw/czsl_repro \
    --work /var/tmp/czsl_xhw_bf_20261008 \
    --seed 3351 --allow-reconstructed
)
```

准备过程：验证原代码 SHA256 → 复制源码到独立目录 → 打上已测试补丁 → 重新编码固定逻辑 bank → 从旧 A/E 配置生成四组 → 冻结配置、知识、划分哈希。不会覆盖 `/src/KCC0601_v1`。

源码守卫若报 `Source version differs...`，先保存输出核对服务器版本。**不要把守卫删掉来强行运行，也不要回退或覆盖原源码。** 这时尚不能认为服务器版本已经适配成功。

## 9. 真实图像 CUDA 冒烟：必须先通过

这一步没有 optimizer step，也不训练模型。只取 TRAIN 的首批 8 张图，检查初始化、逻辑梯度、关闭恒等和推理恒等。为了可比，恒等检查使用 eval 模式关闭 dropout，然后显式调用 `train_forward()`。

```bash
(
  set -euo pipefail
  BF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
  BF_WORK=/var/tmp/czsl_xhw_bf_20261008
  BF_GPU=0
  for BF_VARIANT in B F; do
    CUDA_VISIBLE_DEVICES="$BF_GPU" "$BF_PY" "$BF_WORK/src/tests/smoke_bf.py" \
      --cfg "$BF_WORK/experiments/configs/${BF_VARIANT}_s3351.yml" \
      --out "$BF_WORK/experiments/smoke/${BF_VARIANT}_s3351.json"
  done
)
```

成功输出 `REAL_CUDA_SMOKE_PASSED`。查看报告：

```bash
sed -n '1,100p' /var/tmp/czsl_xhw_bf_20261008/experiments/smoke/B_s3351.json
sed -n '1,100p' /var/tmp/czsl_xhw_bf_20261008/experiments/smoke/F_s3351.json
```

每项 OI/OE/AE 必须对原生视觉投影有有限、非零梯度；`beta0_training_identity`、`fixed_weight_inference_identity`、`parameter_count_equal` 都应为 true。若任一失败，不进入完整训练。已有 smoke 报告不覆盖；重复调用会拒绝，正确完成后直接进入下一步。

## 10. 四组完整训练与验证

以下是本轮实际训练命令。按 A_ref → B → E_ref → F 顺序运行，同一块空闲 GPU，保持环境一致。GPU 物理编号不同就只改 `BF_GPU`。

```bash
(
  set -euo pipefail
  BF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
  BF_BUNDLE=/tmp/logiczsl_cpf_bf_20261008
  BF_WORK=/var/tmp/czsl_xhw_bf_20261008
  BF_GPU=2
  for BF_VARIANT in A_ref B E_ref F; do
    "$BF_PY" "$BF_BUNDLE/run.py" \
      --work "$BF_WORK" --seed 3351 \
      --variant "$BF_VARIANT" --gpu "$BF_GPU" --python "$BF_PY"
  done
)
```

如果 A_ref/E_ref 已按本包成功完成，只启动两项新组的准确命令是：

```bash
/data/xhw/miniconda3/envs/cpf/bin/python /tmp/logiczsl_cpf_bf_20261008/run.py \
  --work /var/tmp/czsl_xhw_bf_20261008 --seed 3351 --variant B --gpu 0

/data/xhw/miniconda3/envs/cpf/bin/python /tmp/logiczsl_cpf_bf_20261008/run.py \
  --work /var/tmp/czsl_xhw_bf_20261008 --seed 3351 --variant F --gpu 0
```

不要调用旧 `tools/run_mvp.sh` 来启动 B/F，它仍是 ACDE 的脚本。

每组完成后得到 `status.txt = train_and_val_complete`；checkpoint 由每轮 VAL CW AUC 的第一个严格最大值选出；随后用同一个 checkpoint 评价 VAL CW/OW。不用 OW 重新选轮次，不读取 TEST 指标。

B/F 训练日志中应出现 `BF_LOGIC epoch=... cpf=... oi=... oe=... ae=... weighted=...`，不只是 `kcc_logic: true`。TensorBoard 也记录这些损失。数值非零、梯度非零只证明接通，不能代替泛化收益。

若任务失败，已产生目录和日志会保留，脚本不会静默重跑/覆盖。保留错误，再在另一个版本工作目录重试；不要一口气删除旧工作目录。若只是四组中的一组尚未启动，单独用对应 `--variant` 运行，不要重跑已经完成的组。

## 11. 汇总真实结果到 Markdown 并下载

```bash
(
  set -euo pipefail
  BF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
  BF_WORK=/var/tmp/czsl_xhw_bf_20261008
  "$BF_PY" /tmp/logiczsl_cpf_bf_20261008/summarize.py \
    --work "$BF_WORK" --seeds 3351 \
    --out "$BF_WORK/experiments/BF_seed3351_实际训练与结果.md"
)
```

同时生成 `.csv`。未完成的组列为 incomplete，不补造指标；但完整比较需要四组全部完成。重复汇总需换输出文件名，以免覆盖已交付的报告。

在 **Windows PowerShell** 下载：

```powershell
$BFResultDir = 'C:\Users\llenovo\Documents\ChatGPT\CZSL\output\BF_server_results_20261008'
New-Item -ItemType Directory -Path $BFResultDir -Force | Out-Null
scp 'xhw@211.87.232.122:/var/tmp/czsl_xhw_bf_20261008/experiments/BF_seed3351_实际训练与结果.md' $BFResultDir
scp 'xhw@211.87.232.122:/var/tmp/czsl_xhw_bf_20261008/experiments/BF_seed3351_实际训练与结果.csv' $BFResultDir
```

每个 run 都保留实际配置、协议、源码快照、环境、知识 manifest、训练日志和 checkpoint 哈希。模型权重保留在服务器；不要只下载小表后删除训练目录。

## 12. 后续多 seed：第一轮之后再执行

若 seed3351 未暴露接口/数值错误，则固定同一份知识、β=0.01、q=3，以相同流程补 seed0/2026。不要根据初轮结果改知识或 β 后混成同一组均值。

```bash
(
  set -euo pipefail
  BF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
  BF_BUNDLE=/tmp/logiczsl_cpf_bf_20261008
  BF_WORK=/var/tmp/czsl_xhw_bf_20261008
  BF_GPU=0
  for BF_SEED in 0 2026; do
    "$BF_PY" "$BF_BUNDLE/prepare.py" \
      --root /data/xhw/czsl_repro --work "$BF_WORK" \
      --seed "$BF_SEED" --allow-reconstructed
    for BF_VARIANT in B F; do
      CUDA_VISIBLE_DEVICES="$BF_GPU" "$BF_PY" "$BF_WORK/src/tests/smoke_bf.py" \
        --cfg "$BF_WORK/experiments/configs/${BF_VARIANT}_s${BF_SEED}.yml" \
        --out "$BF_WORK/experiments/smoke/${BF_VARIANT}_s${BF_SEED}.json"
    done
    for BF_VARIANT in A_ref B E_ref F; do
      "$BF_PY" "$BF_BUNDLE/run.py" --work "$BF_WORK" \
        --seed "$BF_SEED" --variant "$BF_VARIANT" --gpu "$BF_GPU" --python "$BF_PY"
    done
  done
  "$BF_PY" "$BF_BUNDLE/summarize.py" --work "$BF_WORK" \
    --seeds 3351 0 2026 --out "$BF_WORK/experiments/BF_三seed_实际训练与结果.md"
)
```

这是额外 8 次训练，不是初轮必须立刻消耗的预算。三 seed 汇总报告自动记录配对差值均值；正式结论还应看每个 seed 的方向、HM/OW 退化、方差和知识版本，而不只看平均 AUC。

## 13. 如何判断结果

- B − A_ref 是最直接的“原 CPF + 外部逻辑损失”检验。
- F − E_ref 检验文本/校准路线上的逻辑增量；F − A_ref 检验整体是否优于 CPF。
- F − E_ref 为正但 F − A_ref 仍为负：逻辑改善了 E，但整体仍被旧校准拖累，不能写“方法整体超过 CPF”。
- 单 seed 没收益：先看逻辑梯度、加权项量级、描述区分性与名称/描述 softmax 分布；不能马上断言所有知识都无效。
- 单 seed 有收益：固定协议补多 seed，而不是直接换视觉框架或加随机图。
- 当前 VAL 已被前期实验反复用于诊断，因此本轮 VAL 属于开发证据；最终配置冻结后才能安排独立 TEST 报告，不在 TEST 上挑 β/知识/seed。

这轮暂不改 λ、不改 backbone、不加 EKR，不同时添加知识节点辅助 CE。后续如果研究 λ=0 的文本+逻辑，需要单独命名 `C+Logic`/`F_noCal`，不是覆盖原 F。

## 14. 本地验证记录与限制

`local_cpu_test.json` 记录 13 项 CPU/流程检查结果：作者公式数值一致；五类 logits 有有限非零梯度；零违反度梯度稳定；类别顺序对齐；每项逻辑连接视觉表示；β 只乘一次；无新增参数；使用实际 CPF 前向 AST 配合张量替代 backbone 的训练/推理恒等检查；安装不改原源码；准备、协议和配对汇总流程检查，以及 Python 3.8 语法检查。

真实 GloVe bank 验证得到 5×300 父类、12×3×300 OE、16×3×300 AE，全部词覆盖，见 `local_glove_check.json`。它不需要额外图像数据，服务器仍重新编码并记录自己的文件哈希。

原归档环境为 Torch 1.13.1+cu117、NumPy 1.24.4；新代码保留 Python 3.8 语法，不要求升级原环境。知识 bank 的安全 tensor-only 加载参数已对照 [PyTorch 1.13.1 官方源码](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/serialization.py)核实。当前本地数值测试实际运行于 Torch 2.5.1 CPU，不能代替原服务器环境的运行验证。

**限制仍然存在**：重建知识不是历史原数据；GloVe 短语均值不等于 CLIP 文本表示；材料属性视觉可辨性有限；OI 分组需要审核；CPU 张量测试不等于真实 CUDA 前向；服务器版本需要守卫校验；目前没有任何新 B/F 完整训练指标。

## 15. 来源与可追溯性

- 本地原论文：[用户提供的 LOGICZSL PDF](C:/Users/llenovo/Desktop/CZSL/Wu_LOGICZSL_Exploring_Logic-induced_Representation_for_Compositional_Zero-shot_Learning_CVPR_2025_paper.pdf)。核对 §3.2-3.4、Fig.4、Eq.6/9/12/13，以及 §4.1。PDF 第4/5页另做了渲染查看。
- [CVPR 官方论文入口](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_LOGICZSL_Exploring_Logic-induced_Representation_for_Compositional_Zero-shot_Learning_CVPR_2025_paper.pdf)。网页抓取返回 403，本次实际依据本地 PDF，不冒称抓到了额外附录。
- [作者公开仓库](https://github.com/Pieux0/LOGICZSL)。2026-10-08 核查，页面列出的根文件为 assets、LICENSE、README；README 说明生成概念知识 JSON，但没有恢复本地缺失的历史知识输出。
- 本地作者源码：`C:/Users/llenovo/Desktop/CZSL/CZSL_code/LogiCZSL_copy.tar.gz`。仅用于实现审计和公式核对，不把压缩包命名当作代码已正确运行的证据。
- CPF 适配参考：2026-10-05 归档的 `files/src/KCC0601_v1`，以及原 A/E 的 effective/requested 配置；守卫哈希固化在 `install.py`。

交付总结：现在已有一个可部署、经过本地检查的 **CPF B/F 方法重建执行包**；真实 CUDA 和完整训练需要你在已有服务器会话按第8-11节执行，结果会自动整理为 Markdown。没有恢复或伪造原作者遗失数据，也没有宣称尚未运行的训练成功。
