# UT归因复核输出：解读与v2修复

日期：2026-10-06。对应 v1 审计目录 `semantic_gate_audit_20261006_144404`。

## 1. 这份输出说明什么

**研究筛选确实未通过，但技术复核尚未完成；这些FAIL目前不能作为删除结果或重新训练的依据。** v1复核脚本有两个已复现的兼容性漏项，下面修正审计工具后再读技术结论。

| seed | C_text − A_unit CW AUC | C_text − C_shuffle CW AUC |
| --- | ---: | ---: |
| 0 | +0.508647 pp | −2.088948 pp |
| 3351 | +4.662171 pp | +1.019615 pp |
| 2026 | +2.125442 pp | +0.269668 pp |
| 平均 | +2.432087 pp | −0.266555 pp |

三次训练中，加描述组都优于只归一化名称组。正确描述对打乱描述则两次略好、一次明显更差，平均没有优势，低于原先+0.5 pp门槛。因此当前仍无法把收益归因于描述的正确归属。三seed和一次固定置换不足以证明稳定性或统计显著性；seed0的负结果应完整保留。

`PASS report_sources_and_gate` 说明报告所记录的来源文件哈希及差值/筛选重算吻合。`PASS frozen_bank_and_shuffle_integrity` 说明冻结bank、原知识文件哈希、张量聚合及置换结构检查通过，不代表描述内容已经人工核实正确。

`FAIL matched_configs_across_variants_and_seeds` 是前面多项run核对提前失败后的连带结果，不能单凭它断言组间学习率或训练预算不同。最佳轮次只列旧A/C的第6轮，是因为其他run提前退出了复核，并不说明新实验还没训练完。

`AUDIT_EXIT_CODE=2` 表示复核工具记录了失败项；此次没有启动训练，也不是训练卡住。

## 2. v1脚本漏掉了什么

### 2.1 日志打印与YAML保存的类型不同

原训练器使用 `print(cfg)` 写日志；包装器用 `cfg.dump()` 写 `effective_config.yml`。同一份配置也可能得到：

| 实际配置值 | 日志打印解析后 | dump文件解析后 |
| --- | --- | --- |
| 浮点数 `1e-05` | 字符串 `'1e-05'` | 浮点数 `1e-05` |
| Python `None` | 字符串 `'None'` | Python `None` |

这个差异已使用原环境记录中的 `yacs==0.1.8` 在本地复现。v1直接比较两份解析字典，把相同值的不同表示误判为不一致。新实验全部停在这一处，符合这种兼容性问题；服务器具体出现了哪些字段差异，将由v2列出，不能只凭截图认定所有配置完全一致。

v2用同一YACS打印器从有效配置重新生成预期的日志表示，再严格比较所有字段。它不跳过配置检查、不吞掉真实参数差异。真正不同的学习率等字段会继续FAIL，并显示具体差异。跨组比较同样统一打印表示，处理旧日志和新YAML之间的格式差别。

### 2.2 旧A/C保存的是bias=0的top5

已查过旧评价器及本地原C输出：旧 `labels.npz` 字段为 `attr/obj/pair/top5_indices_at_bias0`，新评价器保存 `top1_at_bias0`。v1只支持新字段，因此读到复用的旧C结果时出现KeyError。

v2兼容两种格式：新格式直接读取top1；旧格式取保存的排序top5第一列。无需修改原npz或重新评价。配对前仍严格核对图像顺序、候选顺序与真值。精确同分时，旧topk和新argmax的tie处理可能不同，错误分布仍只作为bias=0诊断，不能重算AUC/HM或代替正式偏置扫描指标。

这两个漏项来自我提供的v1复核工具。原先的测试没覆盖真实YACS打印/dump差异以及旧预测文件格式；v2已补充这些用例。

## 3. 具体操作：只重跑复核，不重训

保留v1审计目录及原run、bank、报告、gate。v2工具另装在 `tools/semantic_gate_audit_20261006_v2`，不改动 `src/EKR0601_transition_v1`，避免破坏已有训练代码哈希。

本地PowerShell上传修正版：

```powershell
scp "C:\Users\llenovo\Documents\ChatGPT\CZSL\semantic_gate_audit_20261006_v2.zip" xhw@211.87.232.122:/data/xhw/czsl_repro/tools/
```

在服务器执行整块：

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  TOOL="$ROOT/tools/semantic_gate_audit_20261006_v2"
  REPORT="$EXP/reports/semantic_ut_3seed_diskretry_20261006_141852"
  GATE="$EXP/protocol/semantic_gate_ut_diskretry_20261006_141852.json"
  test -s "$ROOT/tools/semantic_gate_audit_20261006_v2.zip"
  test -s "$REPORT/runs.csv"
  test -s "$GATE"
  if [ ! -d "$TOOL" ]; then
    "$CPF_PY" -m zipfile -e "$ROOT/tools/semantic_gate_audit_20261006_v2.zip" "$ROOT/tools"
  fi
  (
    cd "$TOOL"
    sha256sum -c SHA256SUMS
  )
  "$CPF_PY" -I -c 'import yacs,torch,numpy,yaml; print("AUDIT V2 ENV OK")'
  STAMP=$(date +%Y%m%d_%H%M%S)
  AUDIT="$EXP/diagnostics/semantic_gate_audit_v2_${STAMP}"
  mkdir -p "$EXP/launch_logs"

  set +e
  "$CPF_PY" -I -u "$TOOL/audit_semantic_gate.py" \
    --root "$ROOT" --project "$PROJ" --exp "$EXP" --legacy-exp "$LEGACY" \
    --report "$REPORT" --gate "$GATE" --out "$AUDIT" \
    2>&1 | tee "$EXP/launch_logs/semantic_gate_audit_v2_${STAMP}.log"
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
  exit "$RC"
)
```

CPU即可，时间主要花在读取权重校验哈希，可预留1—10分钟，无需管理员权限。用现有cpf环境，不安装或升级服务器依赖。

## 4. 修正版输出如何判断

- `CONFIG_PRINT_COMPAT ... TRAIN.lr_encoder` 等：预期的打印/保存格式差异已经识别，不是参数自动修改。
- `INTEGRITY_PASSED True`：这一轮技术复核全部通过，再继续原第六步的人工描述审阅。
- `RESEARCH_GATE_PROCEED False`：即使技术检查通过，原先的负研究筛选也应保持false。审计工具修复不会提高AUC或改变筛选规则。
- 若仍出现 `Actual logged/effective config mismatch`，错误中会列出真正不同的字段；先根据这份差异定位问题，不自动重训。
- 若出现权重哈希、训练轮次、评价身份或bank不一致等其他FAIL，保留完整输出再分析。这些检查在v2仍然执行。

技术复核未完成之前，不根据当前FAIL删结果，也不启动六项追加置换训练或正式EKR训练。

## 5. 核验范围

v2在CPU通过10项测试，包括真实YACS序列化、真实旧npz字段样例、混合新旧结果的端到端复核，以及实际参数变化、损坏权重、错误置换、样本错位和重复身份的拒绝检查。Bash命令做过语法检查。未连接服务器运行v2；实际技术结论以新的审计输出为准。
