# 截至 A/C 跨数据集验证：实验报告记录与分析流程

编制日期：2026-10-04。对应用户提供的《第四步补充_lambda0后扩展AC操作.md》。今晚结果尚未读取；本文提供可实际运行的汇总工具与报告写法，不提前填入 MIT/C-GQA 新结果。

## 1. 今晚完成后，这份报告应覆盖什么

报告题目建议：**“CPF 基线复现与知识文本/兼容先验的阶段验证”**。

完成当前流程后，应有三组材料：

| 材料 | 规模 | 报告用途 |
|---|---|---|
| 历史 CPF baseline | 三数据集已归档的 test 结果 | 交代复现基础，独立附表 |
| 当前 MVP | UT 的 A/C/D/E；MIT 和 C-GQA 的 A/C，共 8 个 run | 同预算消融、跨数据集 A/C 主结果 |
| 固定 checkpoint 的 lambda 诊断 | Dpost/Epost × 5 个 lambda × CW/OW = 20 次评价 | 解释为何选择 lambda=0，保留负结果 |

当前 MVP 的每个 run 同时有 validation CW/OW 两份结果，因此预期 16 份主 metrics；CW/OW 不是两个独立训练重复。lambda 诊断也不是重新训练的 D/E。

不能写成“完整 0601 方案验证完成”：目前只实现并跑过 A/C/D/E 的 MVP，其中新增跨数据集部分只扩展 A/C。EKR、USG 和其他模型变体属于下一步设计。

## 2. 先建立唯一可信的数据入口

**正式结果以 `metrics.json` 为准，截图用于复核，不作为主数据源。**

当前工具已生成：

```text
runs/<run_id>/
  status.txt                           # train_and_val_complete 才算完成
  config_requested.yml                 # 请求的训练配置
  command_train.txt
  start_time.txt / end_time.txt
  train.log                            # 训练与逐 epoch 验证
  meta/
    source.sha256 / source_snapshot.tar.gz
    checkpoint.sha256
    dataset_hashes.txt
    cache_manifest.json
    pip_freeze.txt / nvidia-smi.txt
  eval_val_cw/ 和 eval_val_ow/
    metrics.json                       # 完整精度指标及身份/hash
    config_eval_effective.yml           # 实际评价配置
    candidates.json                    # 候选顺序
    image_ids.json                     # 评价图像顺序
    labels.npz                         # GT 与 bias=0 的 top-5
```

`metrics.json` 已有 run_id、dataset、variant、seed、phase、setting、lambda、best_epoch、checkpoint/config/source/knowledge/data 哈希，以及评价样本数、候选数、时间和显存。无需从终端手工抄每个小数。

### 2.1 记录的单位

- `metrics_raw` 的 AUC、HM、accuracy 是 `[0,1]`。汇总表乘 100 **一次**，成为 `%`。
- `C−A` 记录为百分点 `pp`。例如 A=39.470%、C=44.262%，增益约 4.791 pp，以未四舍五入的原数计算。
- 相对提升是 `(C−A)/A×100%`，可选附列；不能把相对百分比与百分点混用。C-GQA 的 AUC 较小，相对提升可能很大，主报告优先绝对 pp。
- `best_seen/best_unseen` 是曲线各自最大值，未必位于同一个 bias。
- `hm_seen/hm_unseen` 才是 best-HM 点对应的两项，可用来核对 `HM=2SU/(S+U)`。
- `closed_attr_match/closed_obj_match` 沿用原evaluator的定义，汇总列名带 `_evaluator`；不能默认它们就是bias0或best-HM点的原语准确率。
- `peak_eval_gpu_allocated_mb` 在现有工具中按 `1024²` 换算，实际是 MiB 的 PyTorch allocated 峰值，不是整卡最大占用。
- `end_time−start_time` 包含训练和两次评价，表名写 `train_plus_val_seconds`，不冒称纯训练时长。

### 2.2 每个结果必须同时记录的身份

```text
dataset + seed + variant + phase + setting
score_mode + lambda_k + best_epoch + checkpoint_hash
code/source_hash + data_hash + knowledge/cache_hash
training_budget + requested_config + effective_eval_config
status + source_path
```

只记录四个指标而没有这些身份信息，后面无法判断是否混用了不同 checkpoint、知识版本或测试设置。

### 2.3 知识文件也要作为实验资产记录

知识结构已补入[报告填写模板](C:/Users/llenovo/Documents/ChatGPT/CZSL/阶段实验报告_截至AC跨数据集验证_填写模板.md)第5节；完整schema、真实示例和缓存构造见[知识文件结构与模型使用说明](C:/Users/llenovo/Documents/ChatGPT/CZSL/cpf_stage_report_tools_20261004_v2/知识文件结构与模型使用说明.md)。

```text
<实际冻结知识源目录>/<ut-zap50k|mit-states|cgqa>/
  oi_dict.json       # 父概念 -> 对象列表
  oe_dict.json       # 对象 -> 3条描述
  ae_dict.json       # 属性 -> 3条描述
  attributes.txt / objects.txt
  entry_provenance.csv / review_status.csv
  manifest.json
  K_oa_text_lexical.csv / K_oa_text_report.json

cache_trusted_v1/<dataset>/
  labels.json
  W_o_K.pt [N,300] / W_a_K.pt [M,300]
  K_oa_text.pt / K_oa_prior_centered.pt [N,M]
  W_ao_K/obj_XXXXX.pt [M,300]
  名称向量、report、manifest 等辅助文件
```

| 数据集 | 属性/对象/父概念 | OE/AE描述条数 | provenance/review行数 |
|---|---|---|---:|
| UT | 16 / 12 / 4 | 36 / 48 | 28 |
| MIT | 115 / 245 / 19 | 735 / 345 | 360 |
| C-GQA | 413 / 674 / 23 | 2022 / 1239 | 1087 |

上述基础数量已核对本机draft；最终来源与审核状态由服务器冻结hash核验。C实际用聚合后的W_o_K/W_a_K；OI没有用于当前图传播，pair blocks也未直接消费。不能将C收益写成随机语义图或完整OI逻辑方案的收益。

汇总器从cache `input_sha256`推导实际源目录，保存小文件并核对hash。`approved`与`bulk_trusted_unverified`并列记录，不能因为结构检查通过就写成逐条人工审核。MIT/C-GQA来源为WordNet首义重构，UT为手工重构，非官方LOGICZSL知识。

## 3. 已准备好的工具包

本机文件：

```text
C:/Users/llenovo/Documents/ChatGPT/CZSL/
  cpf_stage_report_tools_20261004_v2.zip
  cpf_stage_report_tools_20261004_v2/
    collect_stage_report.py
    analyze_bias0_errors.py
    plot_stage_report.py
    knowledge_evidence.py
    知识文件结构与模型使用说明.md
    historical_baseline_reference.json
    reference_history/                  # 三份已归档 final_results 原文
    README.md
```

汇总器只读现有实验，不训练、不调用 evaluator、不需要 GPU，也不导入 torch。完整配置核验使用当前 CPF 已有的 PyYAML；缺失时仍可导出，但可比性会标成 `metadata_only`，不能通过最终完整核验。

逐图分析需要 numpy。画图需要 matplotlib，推荐下载报告后使用本机已确认存在的 Anaconda 环境画图，避免为了报告改动服务器训练环境。

## 4. 上传工具包

在**本机 PowerShell**执行：

```powershell
scp 'C:/Users/llenovo/Documents/ChatGPT/CZSL/cpf_stage_report_tools_20261004_v2.zip' 'xhw@211.87.232.122:/data/xhw/czsl_repro/experiments/kcc0601_glove_mvp_v1/'
```

这里使用此前截图中的 SSH 地址和用户名，默认端口 22。若你平时使用其他 SSH 别名或端口，沿用实际登录配置，例如 scp 用 `-P <端口>`；本文没有验证当前连接。

随后在**服务器 Bash**执行一次：

```bash
(
  set -euo pipefail
  EXP=/data/xhw/czsl_repro/experiments/kcc0601_glove_mvp_v1
  TOOLS="$EXP/report_tools_20261004_v2"
  test ! -e "$TOOLS" || { printf 'Tools directory already exists: %s\n' "$TOOLS"; exit 2; }
  unzip -n "$EXP/cpf_stage_report_tools_20261004_v2.zip" -d "$TOOLS"
  cd "$TOOLS"
  sha256sum -c package_files.sha256
  test -s "$TOOLS/collect_stage_report.py"
)
```

如果工具目录已存在，检查原文件后直接复用该目录；不要重复覆盖。上述块在子 shell 中执行，失败会返回提示，不会因为顶层 `set -e` 直接结束 SSH 会话。

## 5. 今晚跑完后，一次生成报告和全部主表

下面在服务器 Bash 执行。先设置变量，这样输出路径可用于后续命令：

```bash
ROOT=/data/xhw/czsl_repro
EXP="$ROOT/experiments/kcc0601_glove_mvp_v1"
TOOLS="$EXP/report_tools_20261004_v2"
CPF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
REPORT="$EXP/reports/stage_$(date +%Y%m%d_%H%M%S)"
```

完整汇总：

```bash
(
  set -euo pipefail
  "$CPF_PY" "$TOOLS/collect_stage_report.py" \
    --experiment "$EXP" \
    --cache-root "$ROOT/knowledge/kcc0601_v1/cache_trusted_v1" \
    --lambda-dir "$EXP/diagnostics/lambda_post_v1" \
    --audit-dir "$EXP/audit/ac_transfer_v1" \
    --out "$REPORT" \
    --require-complete
)
printf 'REPORT=%s\n' "$REPORT"
```

若实际 smoke 使用了 `ac_transfer_v2` 等版本，只修改 `--audit-dir` 为真实路径，不伪造 v1 输出。

知识源目录通常会按cache中的冻结路径自动找到。若目录搬迁，先从下面命令查看记录的原路径：

```bash
"$CPF_PY" - "$ROOT/knowledge/kcc0601_v1/cache_trusted_v1" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for ds in ('ut-zap50k', 'mit-states', 'cgqa'):
    man = json.loads((root/ds/'cache_manifest.json').read_text())
    print('\nDATASET:', ds, 'status:', man.get('knowledge_status'))
    for path, digest in man.get('input_sha256', {}).items():
        if Path(path).name in ('oi_dict.json','oe_dict.json','ae_dict.json','review_status.csv','manifest.json'):
            print(digest, path)
PY
```

如果仍保留同版本源文件、只是迁移到了另一父目录，在完整汇总命令加 `--knowledge-root "/真实父目录"`（其下应有三个数据集子目录），换新REPORT重跑。该选项仍核对冻结hash，不允许以另一个draft替代训练来源。缺失原源文件时保留报告为待核验状态，不修改训练缓存。

期望最后出现：

```text
VERIFIED MVP EVALUATIONS: 16/16; lambda rows: 10/10; issues: 0
COMPLETE: True
```

如果不是 True，报告仍会保存，但 `--require-complete` 返回非零状态。先查看：

```bash
"$CPF_PY" - "$REPORT" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
print(json.dumps(json.loads((root/'collection.json').read_text()), ensure_ascii=False, indent=2))
with (root/'tables/issues.csv').open(encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        print(row)
PY
```

缺少 run/metrics 时等待对应任务完成；hash、身份、配置不一致时查原文件。不能为了通过核验手工改 status、修改指标或删除 hash 字段。修复后用新的 `REPORT` 时间戳重新汇总，保留旧草稿。

### 5.1 训练未全部结束时

可以去掉 `--require-complete` 导出一次中期快照，输出目录用新版本。未完成的结果不会进入有效 A/C 配对表。正在写入的日志会标记为临时证据。

报告生成通常是 CPU 文件读写，明显小于训练成本，不需要占 GPU。日志较大或共享盘慢时可能需要几分钟；汇总和归档可以放在现有 tmux 会话中，无需新建训练任务。

### 5.2 生成的文件

| 文件 | 重点看什么 |
|---|---|
| `阶段实验报告_自动草稿.md` | 自动插入真实结果的阶段报告基础稿 |
| `tables/run_inventory.csv` | 8 个 run 是否完成，最佳 epoch、耗时 |
| `tables/all_val_metrics.csv` | 16 份完整指标，含原始身份/hash和数据状态 |
| `tables/ac_comparison.csv` | 三数据集 × CW/OW 的 A/C 配对与 pp 增益 |
| `tables/pair_audit.csv` | 配置与数据来源是否允许直接比较 |
| `tables/ut_acde_effects.csv` | C−A、D−A、E−C、E−D、E−A、交互差值 |
| `tables/lambda_analysis.csv` | 10 个候选的数值、约束失败原因、选择解释 |
| `tables/epoch_curves.csv` | 收敛/欠训练/过拟合线索 |
| `tables/knowledge_summary.csv` | 描述 OOV、K 分布与审核状态 |
| `tables/knowledge_structure.csv` | 实际OI/OE/AE数量、来源路径、审核条目与hash核验状态 |
| `knowledge_schema.json` | JSON示例、CSV字段、来源状态及逐文件hash；矩阵形状为manifest合同 |
| `evidence/knowledge_source/` | 与训练cache追溯核验的知识源文件快照 |
| `evidence/documentation/知识文件结构与模型使用说明.md` | 知识源目录、缓存目录和模型实际消费说明 |
| `tables/historical_baseline_test_reference.csv` | 独立历史 test 附表，来自打包原文 |
| `tables/issues.csv` | 缺失与不一致；不应忽略 |
| `evidence/` + `evidence_manifest.json` | 原始小文件快照与校验值 |
| `report_files.sha256` | 本次汇总产物的校验清单 |

完整精度 CSV 用于计算，报告展示通常保留 3 位小数。每个 dataset 目前只有一对 seed，不生成伪造的 std、误差棒或显著性符号。

## 6. 逐图分析：C 到底改对了哪些、改错了哪些

建议首先做 OW 的 bias=0 分析。在服务器 Bash：

```bash
(
  set -euo pipefail
  "$CPF_PY" "$TOOLS/analyze_bias0_errors.py" \
    --report "$REPORT" \
    --data-root "$ROOT/data" \
    --setting ow \
    --out "$REPORT/bias0_ow_v1"
)
```

如需 CW，另用 `--setting cw --out "$REPORT/bias0_cw_v1"`。输出包括：

- `bias0_summary.csv`：全部/seen/val-unseen 的固定 bias=0 原语与组合准确率。
- `<dataset>_image_cases.csv`：每图 GT、A/C 预测、seen 组别和变化类型。
- `per_attribute_bias0.csv`：每属性样本量与组合准确率变化。
- `issues.csv`：不能对齐的数据集及原因。

工具会核对图像顺序、候选顺序、GT、原语词表，以及 `train_pairs.txt` 的哈希，不靠重新枚举图片目录配对，也不读取 test pair 列表做归因。

重点记录四类数量：

```text
两者都正确
A错、C对：知识帮助修正
A对、C错：知识引入错误
两者都错误
```

净修正量：`A错C对 − A对C错`。在同一个固定 operating point，除以图像数再乘100等于组合准确率的 pp 差值。

再看：对象准确率是否上升？属性准确率是否上升？在对象已判断正确的案例中是否修正了属性？增益集中在哪些属性、哪些对象/组合？每类同时报告样本数，小样本大涨幅不当作稳定证据。

**限制：此分析是 bias=0，而主 AUC/HM 来自偏置扫描。** 若 bias0 准确率不升而 AUC 上升，并不自动矛盾；它们度量不同的曲线/阈值。不能从 top-5 重建 AUC，也不能从原始 cosine 当作概率计算 ECE。

若当前没有保存完整 `scores.dat`，可做上述错误分析，但不能事后生成可信的完整 seen/unseen 曲线或 best-HM 点逐图预测；如未来需要，在独立版本评价目录使用现有 `--save-scores` 选项保存全部分数。C-GQA OW 的磁盘开销应先按 `4*n_images*n_candidates` 估算，不能为写报告直接盲目开启。

## 7. 主结果应怎么分析

### 7.1 三数据集 A/C：当前最重要的问题

逐数据集、逐 CW/OW 写明：AUC、HM、best-seen、best-unseen、HM 点的 S/U 及 C−A。

推荐分析顺序：

1. C 的 CW AUC 是否高于 A？这是当前预设主指标。
2. HM 是否同方向？若 AUC 提高但 HM 降低，说明整条折中曲线与最佳平衡点收益不一致。
3. OW 是否同方向？OW 与 CW 分开，不能只挑更好的一项。
4. unseen 是否改善？同时看 seen 的代价和 HM 点 S/U，不能只报各自最大 unseen。
5. 三数据集是否一致？明确列出持平/失败的设置，不能用平均值掩盖。

可以写“在三个数据集各自单 seed 的 validation 对照中，C 在 X/Y 个设置上获得正向 AUC 差值”。不能写“重复六次均有效”：CW/OW 共享 checkpoint，且三数据集不是同一实验的独立 seed 重复。

### 7.2 UT A/C/D/E：文本与校准分别贡献什么

把它理解为一个描述性的 2×2 模块表：

| | 不加校准 | 加当前固定 K 校准 |
|---|---|---|
| 无新文本 | A | D |
| 有新文本 | C | E |

记录 `C−A`（文本）、`D−A`（校准）、`E−C`（文本存在时的校准）、`E−D`（校准存在时的文本）、`E−A`（总效果）、`E−C−D+A`（描述性非加性差值）。

当前截图中，C 提升而 D/E 下降。准确措辞是：**该知识版本、legacy score、lambda=0.1、单 seed MVP 下，文本路径获益，当前校准路径损害指标。** 不能据此声称概率框架或所有兼容知识普遍失败。

各模型按自己的 validation-best epoch 选 checkpoint。该比较回答“给定训练与选择协议的最终效果”，不是同一 epoch 的纯模块因果效应。原设计中的正式多 seed 或更多变体仍未完成。

### 7.3 Lambda：为什么选择0

报告完整网格 `{0,0.025,0.05,0.1,0.2}`，Dpost 基于 A checkpoint，Epost 基于 C checkpoint。

原规则：最大化 val CW AUC；CW HM 下降不超过0.5 pp；OW AUC下降不超过0.5 pp；OW best-unseen下降不超过2 pp；同主指标选更小lambda。

对每个未选候选分别说明：主指标更低、保护约束失败、或主指标相同但lambda更大。即使某一项有提升，只要未满足采用规则，也要如实列出，不能统一写成“全都负收益”。

lambda=0 必须复现对应 A/C 的指标与 checkpoint。复现失败优先检查评价一致性，不能继续解释 K 的有效性。

### 7.4 学习曲线：是否还有训练预算问题

用 `epoch_curves.csv` 看 A/C 以及 UT D/E 的训练 loss 和 validation AUC。

- 最佳 epoch 恰好在第10轮，且曲线仍上升：可能预算不足，但不是继续训练一定有效的证据。
- train loss继续下降而val AUC下降：可能过拟合或选型波动，需多 seed 验证。
- C 比 A 更早达到较高val：可以描述收敛差异，不能仅凭单 seed称“训练效率普遍提升”。
- 若要延长训练，A/C使用同样追加预算、新protocol/run ID；不能只延长失败的一侧，或将10轮A与20轮C混成主表。

现有训练日志只打印四位小数的 validation，曲线精度约0.01 pp。正式差值仍以完整 JSON 为准。现有流程通常仅对最佳 checkpoint 做 OW，因此没有 OW 每轮曲线时留空，不补造。

## 8. 画图与下载归档

### 8.1 先下载完整报告

服务器 Bash：

```bash
(
  set -euo pipefail
  test -s "$REPORT/collection.json"
  test ! -e "$REPORT.tar.gz" || { printf 'Archive already exists: %s\n' "$REPORT.tar.gz"; exit 2; }
  tar -czf "$REPORT.tar.gz" -C "$(dirname "$REPORT")" "$(basename "$REPORT")"
  sha256sum "$REPORT.tar.gz"
)
printf 'DOWNLOAD=%s.tar.gz\n' "$REPORT"
```

在本机 PowerShell，用服务器刚输出的**实际目录名**替换一次 `$taskReportName`：

```powershell
$taskReportName = 'stage_实际时间戳'
$taskReceived = 'C:/Users/llenovo/Documents/ChatGPT/CZSL/received_reports'
New-Item -ItemType Directory -Path $taskReceived -Force | Out-Null
scp "xhw@211.87.232.122:/data/xhw/czsl_repro/experiments/kcc0601_glove_mvp_v1/reports/$taskReportName.tar.gz" "$taskReceived/"
Get-FileHash -LiteralPath "$taskReceived/$taskReportName.tar.gz" -Algorithm SHA256
tar -xzf "$taskReceived/$taskReportName.tar.gz" -C "$taskReceived"
```

比较本机与服务器的 tar.gz 哈希。报告目录是独立快照，后续训练继续写原 run 不会自动改变已下载报告。

### 8.2 本机生成四组图

已核验本机 `C:/Users/llenovo/anaconda3/python.exe` 有 numpy、PyYAML、matplotlib。PowerShell：

```powershell
$taskReport = "$taskReceived/$taskReportName"
& 'C:/Users/llenovo/anaconda3/python.exe' 'C:/Users/llenovo/Documents/ChatGPT/CZSL/cpf_stage_report_tools_20261004_v2/plot_stage_report.py' --report "$taskReport" --out "$taskReport/figures_v1"
```

输出 PNG（300dpi）和 PDF：三数据集 A/C 增益、UT ACDE 消融、lambda曲线、训练/验证收敛曲线。单 seed 图不加误差棒。英语坐标避免服务器中文字体缺失；报告正文可用中文解释。

画图只接受通过可比性核验的主 A/C 比较。文件或字段缺失时对应图/位置留空，不用0代替缺失。

## 9. 报告正文建议结构

使用同目录的 [阶段实验报告填写模板](C:/Users/llenovo/Documents/ChatGPT/CZSL/阶段实验报告_截至AC跨数据集验证_填写模板.md)，或以自动草稿为数据基础另存提交版。

1. **研究目标**：文本知识与固定兼容先验是否改善现有 CPF。
2. **复现基础**：历史 baseline test 独立表，标出原始资产。
3. **数据与协议**：三数据集、seed、backbone、GloVe、训练预算、候选设置、checkpoint规则。
4. **知识结构、构建与质量**：OI/OE/AE schema及数量、来源/审核CSV、源目录与cache对应、模型实际消费、bulk trust状态、OOV及K分布。
5. **UT消融**：ACDE全结果与各差值，负结果不删除。
6. **lambda诊断**：全部候选、选择约束、零复现及未选原因。
7. **跨数据集验证**：三数据集 A/C，全 CW/OW 指标。
8. **过程与错误分析**：曲线、bias0净修正、成功与失败案例。
9. **结论与限制**：区分已观察效果、待验证原因、下一步假设。
10. **可复现附录**：配置、命令、hash、状态、异常及未完成项。

报告中的一句结论应能对应具体表/图/source。例如“文本知识在 UT 的 CW AUC 提高4.791 pp”有数据依据；“对象条件知识解决属性纠缠”目前只是待验证机制假设。

## 10. 这阶段可以说到哪里

可以说：完成 CPF 的阶段复现、文本/固定K消融、validation-only后处理诊断，以及三数据集各自单 seed 的 A/C工程验证（前提是今晚任务实际完成并核验）。

不能说：多 seed 稳定、统计显著、达到最新 SOTA、全套逻辑/条件特征实验完成，或 EKR/USG 已产生增益。

下一步由实际结果决定：多数据集正向则先做配对多 seed；只UT正向则审查描述与数据域；普遍无增益则冻结当前负结果，重新审视知识输入。任何新路线用新代码/知识/配置版本，不覆盖这一阶段。

