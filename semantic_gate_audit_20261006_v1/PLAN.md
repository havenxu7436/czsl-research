# 第六步：UT 归因门槛未通过后的复核与 EKR 决策

日期：2026-10-06。对应报告 `semantic_ut_3seed_diskretry_20261006_141852`。

**现在先执行第1—4节的只读复核。暂不增加正式训练，也不改原 gate。** 这一步要区分实现/文件问题、描述质量问题，以及在当前知识和训练预算下正确语义没有稳定额外收益这三种情况。随后按第5节选择一个有依据的实验修订。

附带工具：`semantic_gate_audit_20261006_v1.zip`。工具在原训练代码目录之外运行，使用现有 `cpf` 的 Python、torch、numpy、PyYAML，CPU 即可，无需管理员权限，也不需要重装环境。

## 0. 这份结果目前能说明什么

以已经显示的三个配对差值计算：

| 比较 | 三 seed 平均 CW AUC 差值 | 当前解释 |
| --- | ---: | --- |
| C_text − A_unit | +2.4321 pp | 加入描述的方案优于只归一化类别名称 |
| C_text − C_shuffle | −0.2666 pp | 正确描述没有稳定优于当前固定的描述置换 |

这里的 `pp` 是百分点。原门槛要求第二项至少为 +0.5 pp，当前未通过。

原截图的差值数组不直接标注 seed。工具从报告的 `seed` 字段和 gate 的 `planned_seeds` 对应关系读取，明确打印 −2.0889 pp 对应哪一个 seed；不凭截图数组顺序判定。

此结果支持“文本输入的变化有收益”，还不足以支持“正确知识语义是收益来源”。`proceed:false` 是预先约定的投入筛选结果，不是统计检验，也不能证明所有条件知识模型都无效。

目前对照使用一个固定置换（默认属性1729、对象1730）。三次训练 seed 变化不等于三次描述置换变化。后续若需要判断收益对置换是否稳健，必须另外设计配对实验。

## 1. 上传复核工具

在本地 Windows PowerShell 执行，使用此前连接服务器的地址：

```powershell
scp "C:\Users\llenovo\Documents\ChatGPT\CZSL\semantic_gate_audit_20261006_v1.zip" xhw@211.87.232.122:/data/xhw/czsl_repro/tools/
```

在服务器终端执行：

```bash
(
  set -euo pipefail
  ROOT=/data/xhw/czsl_repro
  CPF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
  TOOL="$ROOT/tools/semantic_gate_audit_20261006_v1"
  test -s "$ROOT/tools/semantic_gate_audit_20261006_v1.zip"
  if [ -e "$TOOL" ]; then
    printf '%s\n' 'Tool directory already exists; reuse it if this version was already extracted.'
  else
    "$CPF_PY" -m zipfile -e "$ROOT/tools/semantic_gate_audit_20261006_v1.zip" "$ROOT/tools"
  fi
  test -s "$TOOL/audit_semantic_gate.py"
  cd "$TOOL"
  sha256sum -c SHA256SUMS
  "$CPF_PY" -I -c 'import sys,torch,numpy,yaml; print("AUDIT ENV OK",sys.executable,torch.__version__)'
)
```

预期看到 `AUDIT ENV OK ... 1.13.1+cu117`。若仍出现此前 `pooling` 或 `connectionpool` 模块不可见，先保留错误，处理环境/系统问题；不要用重新训练来处理导入故障。

工具只校验并加载自己的 bank/cache 张量，不加载模型权重进行推理；权重只读取字节计算 SHA256。源代码目录、原知识和实验文件保持原样。

## 2. 执行一次完整复核

这一项检查12个实验身份（四组 × 三 seed），核对24份 CW/OW 验证指标。报告里的其他数据集行不混入 UT 差值计算。

可直接在 SSH 终端运行。主要时间用于读取数 GB 权重计算哈希，通常可预留1—10分钟，取决于磁盘和负载；不占训练 GPU。网络不稳定时用已有 tmux 会话即可，不需要专门挂整夜任务。

下面自动按报告 fingerprint 找对应的 gate。若找到多个 gate 会停止并列出路径，避免选错；这种情况下明确设置 `GATE` 为对应报告的实际文件，再执行脚本调用部分。

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  TOOL="$ROOT/tools/semantic_gate_audit_20261006_v1"
  REPORT="$EXP/reports/semantic_ut_3seed_diskretry_20261006_141852"
  test -s "$REPORT/runs.csv"
  test -s "$REPORT/collection.json"

  GATE=$("$CPF_PY" - "$EXP" "$REPORT" <<'PY'
import json, sys
from pathlib import Path
exp, report = map(Path, sys.argv[1:])
fingerprint = json.loads((report / 'collection.json').read_text())['fingerprint']
matches = []
for path in sorted((exp / 'protocol').glob('semantic_gate*.json')):
    record = json.loads(path.read_text())
    if record.get('report_fingerprint') == fingerprint:
        matches.append(path)
if len(matches) != 1:
    print('STOP: expected one gate matching this report; found ' + str(len(matches)), file=sys.stderr)
    for path in matches:
        print(path, file=sys.stderr)
    raise SystemExit(1)
print(matches[0])
PY
  )
  printf 'MATCHED_GATE=%s\n' "$GATE"
  STAMP=$(date +%Y%m%d_%H%M%S)
  AUDIT="$EXP/diagnostics/semantic_gate_audit_${STAMP}"
  mkdir -p "$EXP/launch_logs"

  set +e
  "$CPF_PY" -I -u "$TOOL/audit_semantic_gate.py" \
    --root "$ROOT" --project "$PROJ" --exp "$EXP" --legacy-exp "$LEGACY" \
    --report "$REPORT" --gate "$GATE" \
    --out "$AUDIT" \
    2>&1 | tee "$EXP/launch_logs/semantic_gate_audit_${STAMP}.log"
  pipeline_codes=("${PIPESTATUS[@]}")
  RC=${pipeline_codes[0]}
  if [ "$RC" -eq 0 ] && [ "${pipeline_codes[1]}" -ne 0 ]; then
    RC=${pipeline_codes[1]}
  fi
  set -e

  printf 'AUDIT_EXIT_CODE=%s\nAUDIT_DIRECTORY=%s\n' "$RC" "$AUDIT"
  if [ -s "$AUDIT/audit_summary.md" ]; then
    cat "$AUDIT/audit_summary.md"
  fi
  if [ "$RC" -ne 0 ]; then
    printf '%s\n' 'STOP: inspect the audit failure. Do not retrain or change the gate yet.'
    exit "$RC"
  fi
  printf '%s\n' 'Audit finished. Integrity pass does not imply research gate pass.'
)
```

脚本返回值：0表示文件/协议复核均通过；2表示完成了复核并记录了失败项；1表示脚本不能正常启动或创建输出。**原研究 gate 为 false 不会单独导致返回2**，它和技术完整性是两个独立结论。

每次执行使用新的时间戳目录。复核工具会拒绝覆盖既有审计结果，也会拒绝将输出写入原代码、run、知识、gate或原报告目录。

### 2.1 自动检查的内容

1. 报告来源文件哈希与原 collection 一致；重算配对差值和九项 gate 条件，核对原 gate。
2. 对每项训练确认 seed、起止轮次、十轮验证记录、完成标记、最佳轮次和权重哈希；确认 CW/OW 使用对应的最佳权重。
3. 核对 GloVe、损失/优化等有效配置中的共同字段；四组允许的差异仅包括文本模式、seed、组名和输出路径等预定字段。
4. 核对新实验记录的代码、训练bank和当前文件一致；旧 seed=3351 的 A/C 单独标记为复用记录，恢复其日志里的有效配置。
5. 验证单位归一化、描述均值和中心化残差；验证 C_text 表与原 C 缓存一致。
6. 验证固定置换无自映射，C_shuffle 的类别名称仍属于自己，描述和有效槽位来自 donor，打乱后的均值与归一化计算一致。
7. 对 C_text 与 A_unit/C_shuffle 的验证样本进行配对：图像顺序、候选顺序及真值必须一致，才统计已见/未见、属性/对象的错误变化。

样本错误使用已经保存的 **bias=0 原始 top1**。它用于定位错误分布，不能替代 AUC/HM，也不能由它推断偏置扫描后的最终收益。工具只消费既有 val 输出和 train pair 定义，不启动 test 评价、不读 test pair 文件，也不查看 test 图像。

## 3. 阅读输出，先形成明确诊断

主要产物：

| 文件 | 用途 |
| --- | --- |
| `audit_summary.md` | seed差值、各项PASS/FAIL、最佳轮次和限制 |
| `audit.json` | 完整数值、输入哈希、描述统计、配对错误分布 |
| `description_review.md` | 按属性/对象逐条列出原描述和 shuffle donor，人工复核 |

读取最近一次审计的重点字段：

```bash
(
  set -euo pipefail
  EXP=/data/xhw/czsl_repro/experiments/ekr0601_transition_v1
  CPF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
  "$CPF_PY" - "$EXP" <<'PY'
import json, sys
from pathlib import Path
files = sorted((Path(sys.argv[1]) / 'diagnostics').glob('semantic_gate_audit_*/audit.json'))
if not files:
    raise SystemExit('STOP: no completed audit JSON found')
path = files[-1]
x = json.loads(path.read_text())
print('AUDIT_FILE:', path)
print('INTEGRITY_PASSED:', x['integrity_passed'])
print('FAILED_CHECKS:', [c for c in x['checks'] if c['status'] != 'PASS'])
print('GATE:', json.dumps(x.get('gate_recomputed'), indent=2))
print('BANK:', json.dumps(x.get('bank_audit'), indent=2))
print('BEST_EPOCHS:')
for r in x['run_audits']:
    print(r['seed'], r['variant'], r['curve']['best_epoch'],
          'boundary=' + str(r['curve']['best_at_budget_boundary']))
print('C_MINUS_SHUFFLE_BIAS0_ERRORS:')
for p in x['paired_bias0']:
    if p['comparison'] == 'C_text-C_shuffle':
        print(p['seed'], p['setting'], 'all=', p['all'],
              'seen=', p['seen'], 'unseen=', p['unseen'])
PY
)
```

依次判断：

- 出现 `FAIL`：先按具体字段检查实现或文件问题。不要先把它解释成方法失败，也不要先删 run 或改学习率。
- 均为PASS但 `RESEARCH_GATE_PROCEED=False`：说明在已核对的实验设置下，这个负结果确实存在。它不会因为“训练完成了”而变成 gate 通过。
- 最佳轮次集中在第10轮：这只是预算可能未覆盖充分收敛的信号。不能仅给 C_text 延长训练，或仅重跑负差值那个 seed；若要扩预算，四组需采用同一预定预算，并使用新协议版本。
- `correct_shuffle_cosine_mean` 较高：可能提示打乱前后平均向量相近，需结合 donor和原描述阅读；没有一个已验证的余弦阈值能直接证明语义失效。
- 存在大量通用句、重复句或 fallback：先人工检查知识。`bulk_trusted_unverified` 和哈希通过说明版本/完整性，不代表文本内容已获人工语义验证。

人工阅读 `description_review.md` 时，对每句检查“是否描述该类别本身、是否提供类别名称以外的可观察信息、是否与其他类别明显混淆”。依据类别语义和训练样本核实，**不要根据某个验证错误类别定向写句子，也不要查看 test 来改知识**。描述条数和标签数以实际文件为准。

该诊断不能只凭一组错误样本断言根因。只看某个 seed 的训练曲线，也不能证明差值完全由随机性造成。

## 4. 导出小型诊断包

先完成第2节，再执行。导出原报告的副本和审计输出，保留原始文件；不打包权重、GloVe、图像或完整分数矩阵，以减少磁盘占用。

```bash
(
  set -euo pipefail
  EXP=/data/xhw/czsl_repro/experiments/ekr0601_transition_v1
  CPF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
  REPORT="$EXP/reports/semantic_ut_3seed_diskretry_20261006_141852"
  AUDIT=$("$CPF_PY" - "$EXP" <<'PY'
import sys
from pathlib import Path
files = sorted((Path(sys.argv[1]) / 'diagnostics').glob('semantic_gate_audit_*/audit.json'))
if not files:
    raise SystemExit('STOP: no audit JSON found')
print(files[-1].parent)
PY
  )
  for name in runs.csv paired_deltas.csv paired_summary.csv collection.json; do
    cp -n -- "$REPORT/$name" "$AUDIT/report_$name"
  done
  GATE=$("$CPF_PY" - "$AUDIT/audit.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['inputs']['gate'])
PY
  )
  cp -n -- "$GATE" "$AUDIT/original_gate.json"
  ARCHIVE="$EXP/reports/$(basename "$AUDIT").tar.gz"
  test ! -e "$ARCHIVE"
  tar -czf "$ARCHIVE" -C "$AUDIT" .
  sha256sum "$ARCHIVE"
  du -h "$ARCHIVE"
  printf 'DOWNLOAD_FILE=%s\n' "$ARCHIVE"
)
```

本地 PowerShell 下载这类审计包：

```powershell
$auditDestination = 'C:\Users\llenovo\Documents\ChatGPT\CZSL\output\semantic_gate_audit_20261006'
New-Item -ItemType Directory -Path $auditDestination -Force | Out-Null
scp 'xhw@211.87.232.122:/data/xhw/czsl_repro/experiments/ekr0601_transition_v1/reports/semantic_gate_audit_*.tar.gz' $auditDestination
Get-ChildItem -LiteralPath $auditDestination -Filter '*.tar.gz' | Get-FileHash -Algorithm SHA256
```

将下载后的SHA256与服务器对应文件的输出核对。`audit_summary.md`、`audit.json`、描述复核和原 `paired_summary.csv` 足以支持下一轮设计讨论。

## 5. 复核之后，怎样继续实验

| 发现 | 下一项操作 | 保留的公平性约束 |
| --- | --- | --- |
| 配置、完成记录、权重或置换不一致 | 定位具体缺陷，保留原run；修正后单独记录受影响实验的技术性重跑 | 不按好坏挑seed；原gate留档，另生成修正报告 |
| 描述有可核实的错误、重复或明显缺少区分信息 | 新建 knowledge_v2/bank_v2，先完整审阅再冻结，重做文本归因 | 四组、三seed、相同预算；旧v1结果仍报告 |
| 文件与语义基本正常，固定置换对比仍有疑问 | 用另外两个预先固定的置换检验当前结果对置换的敏感性 | 三个训练seed全部保留；同时报告原置换，不能取最有利置换 |
| 多置换或修订知识后正确文本仍没有优势 | 限制静态文本路线投入；如继续研究动态EKR，明确作为新的探索协议 | 不把探索结果称为旧gate通过；必须有参数/预算匹配与打乱知识对照 |

### 5.1 推荐的下一项训练预算：先解释固定置换

**仅在复核没有发现技术缺陷、人工知识复核也基本正常时，再实施这项实验。** 先补诊断，不立即展开三个数据集全套 EKR。

在新的置换实验协议中，可提前固定属性置换seed为 `{1729, 2027, 4099}`，对象置换seed分别为属性seed+1。1729是原结果；2027和4099是本次预先规定的两个追加置换，不能先计算验证收益再挑置换。每个置换都保留自己的类别名称，只替换整组描述。

固定训练seed为 `{0, 3351, 2026}`。已有 C_text 和 A_unit 可在原代码、bank正确表、优化设置和评估均一致的条件下复用；追加两种 C_shuffle 各训练三个seed，共 **6项新训练，每项10轮**。C_text在不同置换比较中反复引用，是同一个观测，不能当成更多独立重复实验。

令 \(c_s\) 为 C_text 的 CW AUC，\(q_{s,p}\) 为训练seed \(s\)、置换 \(p\) 的 C_shuffle CW AUC：

\[
\Delta_{s,p}=c_s-q_{s,p},\qquad
\bar\Delta_s=\frac{1}{3}\sum_{p\in\{1729,2027,4099\}}\Delta_{s,p},\qquad
\bar\Delta=\frac{1}{3}\sum_s\bar\Delta_s.
\]

报告完整3×3差值、各置换均值和各训练seed均值。这里的9个差值共享 C 权重，**不能按9个独立样本做显著性检验**。

这项实验用于敏感性分析，不能重新定义原gate使它通过。将完整结果按上述公式汇总；如准备将其作为新探索协议的投入依据，可在追加训练前声明工程筛选：整体 \(\bar\Delta\) ≥+0.5 pp，至少2/3训练seed的 \(\bar\Delta_s>0\)，至少2/3置换的三seed平均差值>0；原 C−A_unit 的 CW HM、OW AUC、OW unseen 守门条件继续报告。这不是原gate的新结果，也不是独立验证：当前假设修订已经参考同一验证集，旧置换的负结果必须保留。

结果若仍接近0或为负，说明不值得仅增加同类静态描述的规模。结果若满足上述探索筛选，可讨论动态EKR小原型，并另行冻结后续实验协议；原工具的正式分支入口仍不因这项分析自动解锁。此处不提供立即启动六项训练的命令，是因为它依赖尚未获得的技术和人工复核结论；不能在根因未定位时先增加投入。

### 5.2 若决定尝试动态 EKR，应怎样声明

动态方法仍有可能有效：它要验证的是“图像和候选对象条件下选择描述残差，是否比静态聚合和同参数适配器更有效”。静态描述平均没有稳定语义优势，不能直接回答这个新问题。

但此次证据不足以把该假设写成确定收益。若复核后选择继续，建立独立的探索版本，注明旧gate未通过及继续的原因，先在UT试验：R0（参数匹配适配器）、R3（描述注意力）、R5（中心化条件残差）、R7（描述置换残差）共同配对。R7用于检查新增分支是否依赖正确知识，不能等R5有利后才补。

原方案的核心可写成：

\[
d_{a,r}=t_{a,r}-\mu_a,\quad
\alpha_{a,r}(x,o)=\operatorname{softmax}_r\left(q(x,o)^\top k_{a,r}/\tau\right),
\]
\[
s_{\rm EKR}(x,a,o)=s_C(x,a,o)
+\eta\sum_r\alpha_{a,r}(x,o)\,u(x)^\top d_{a,r}/2,
\qquad \eta=0.1\tanh(g),\quad g_{\rm init}=0.
\]

\(u(x)\) 是归一化的原CPF属性特征，\(t\) 是归一化名称/描述向量，\(\mu\) 是有效槽位均值；null残差为0，并纳入注意力候选。模型初始与 C 分数一致，必要时可关掉增量。正式探索前需另冻结预算、seed、选择规则和停止规则，并相应实现独立探索配置；不能只把旧gate文件false改成true。

这是“条件文本残差”路线。C_shuffle/R7是知识归属对照，**不是USG随机语义图方法**。USG的随机图结构、采样、图正则等机制继续单独设计和记录。

## 6. 可选：只检查 EKR 接口是否还能正常工作

若第五步中 R0/R3/R5 的 GPU smoke 已通过且代码、bank、C权重均未变化，可跳过本节。`PREFLIGHT PASSED` 和CPU router测试通过不等于这三项真实GPU smoke已通过。

本节允许旧gate为false，因为它只验证接入：每组在两个训练样本上更新三次，不保存正式分支权重，也不进行三轮正式训练。`e3`只是现有 smoke配置的命名。通常是几分钟级别，实际以机器负载为准；需要一张可用GPU。

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  STAMP=$(date +%Y%m%d_%H%M%S)
  SMOKE_OUT="$EXP/audit/ekr_interface_recheck_${STAMP}"
  mkdir -p "$SMOKE_OUT"
  "$CPF_PY" -m transition.cli make-configs \
    --root "$ROOT" --exp "$EXP" --legacy-exp "$LEGACY" --bank-root "$BANK_ROOT" \
    --stage branch --datasets ut-zappos --seeds 3351 --epochs 3 \
    --variants R0 R3 R5 --use-legacy-anchor --smoke-only
  for variant in R0 R3 R5; do
    CUDA_VISIBLE_DEVICES="$GPU" "$CPF_PY" -u -m transition.cli smoke \
      --cfg "$EXP/configs/smoke_ut-zappos_${variant}_s3351_e3.yml" \
      --out "$SMOKE_OUT/${variant}.json" \
      2>&1 | tee "$SMOKE_OUT/${variant}.log"
  done
)
```

每项预期出现 `GPU SMOKE PASSED`：旧C分数一致、初始增量为0、梯度有限且能够更新、C完整状态不变、关闭分支恢复C。它只说明实现可运行，不能说明EKR已训练或有精度提升。若该命令报告已有配置不同，保留输出，不覆盖配置。

## 7. 本次交付的核验范围

命令依据本地 `transition/summary.py`、`configs.py`、`bank.py`、`run.py`、`evaluate.py`、`smoke.py`及原训练器整理。复核脚本的失败检测和模拟端到端测试已在CPU通过，原seed=3351 C训练日志中的十轮曲线/最佳第6轮也用于解析核验。文档Bash代码块已做语法检查。

没有连接服务器执行复核或新训练，最新报告只见于截图。当前实际根因须以服务器审计和人工描述复核为准。工具不重扫完整GloVe、不重新编码每句描述、不在GPU复放全部权重，因此技术PASS的边界是记录和张量运算一致，不能替代语义验证或收益验证。
