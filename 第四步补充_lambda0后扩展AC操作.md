# 第四步补充：Dpost/Epost 均选择 lambda=0 后的操作

日期：2026-10-04。适用结果：UT-Zappos、seed=3351、legacy score、当前 template K；Dpost 和 Epost 的 `selected_lambda` 均为 0，`positive_lambda_selected` 均为 false。

## 1. 当前结论与下一阶段

截图中的结果为：

| 诊断 | 对应模型 | 选择 lambda | val CW AUC% | val OW AUC% |
|---|---|---:|---:|---:|
| Dpost | A checkpoint | 0 | 39.470 | 25.492 |
| Epost | C checkpoint | 0 | 44.262 | 32.079 |

lambda=0 复评与原 A/C 指标一致。本次约束下没有选中正 lambda，下一阶段保留成功的 C 文本知识路径，扩展 A/C 到 MIT-States、C-GQA。

JSON 中 Epost 的 delta=0，是相对于 C 的差值，不是 C 相对 A 的差值。当前 C 相对 A 的 CW/OW AUC 增益仍约为 +4.791/+6.587 pp。

不要从这个选择结果推断“所有非零 lambda 的所有指标都更差”。需要查看完整 CSV 才能区分：主指标未提升、保护指标不达标或相同主指标下更小 lambda 获胜。本次诊断也不证明所有 K 或其他训练协议无效。

后续操作顺序：冻结诊断 → 生成两数据集配置 → smoke → 两数据集 A/C 共四次训练 → 汇总验证集结果。旧 UT 的 D/E（lambda=0.1）作为负结果保留，当前不扩展其训练。

以下 GPU 操作以管理员已确认服务器适合恢复计算为前提。能够读取 `lambda_selection.json` 不单独证明此前系统隐藏问题已经解决。

## 2. 保存完整诊断和分支决定

先阅读十行候选结果，记录决定并冻结二十份候选 CW/OW metrics 的校验值。此步骤不使用 GPU，无需单独开 tmux。

```bash
(
    set -euo pipefail
    source /data/xhw/miniconda3/etc/profile.d/conda.sh
    conda activate cpf
    ROOT=/data/xhw/czsl_repro
    EXP="$ROOT/experiments/kcc0601_glove_mvp_v1"
    DIAG="$EXP/diagnostics/lambda_post_v1"
    mkdir -p "$EXP/protocol"

    python - "$DIAG" "$EXP/protocol" <<'PY'
import csv, hashlib, json, sys
from datetime import datetime
from pathlib import Path

diag, protocol = map(Path, sys.argv[1:])
selection = json.loads((diag/'lambda_selection.json').read_text())
rows = list(csv.DictReader((diag/'lambda_post_results.csv').open()))
lambdas = [0.0, 0.025, 0.05, 0.1, 0.2]
expected = {(label, lam) for label in ('Dpost', 'Epost') for lam in lambdas}
actual = {(row['diagnostic'], float(row['lambda'])) for row in rows}
if len(rows) != 10 or actual != expected:
    raise ValueError('lambda grid CSV incomplete or duplicated')
for label in ('Dpost', 'Epost'):
    if float(selection[label]['selected_lambda']) != 0.0:
        raise ValueError(label + ' does not belong to lambda0 branch')

print('diagnostic lambda CW_AUC OW_AUC delta_CW_AUC eligible')
for row in rows:
    print(row['diagnostic'], row['lambda'], row['cw_auc_pct'],
          row['ow_auc_pct'], row['delta_cw_auc_pp'], row['eligible'])

files = [diag/'lambda_selection.json', diag/'lambda_post_results.csv']
files += [diag/f'{label}_l{lam}_{setting}'/'metrics.json'
          for label in ('Dpost', 'Epost') for lam in lambdas
          for setting in ('cw', 'ow')]
lines = []
for path in files:
    if not path.is_file():
        raise FileNotFoundError(path)
    lines.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path)+'\n')

stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
record = {
    'scope': 'UT-Zappos seed3351 legacy template K, validation diagnostic',
    'decision': 'expand A/C only to MIT-States and C-GQA',
    'selection': selection,
    'keep_old_D_E_lambda_0_1_as_negative_results': True,
    'evaluate_test_for_selection': False,
}
decision = protocol/f'lambda0_branch_{stamp}.json'
decision.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
checksum = protocol/f'lambda_post_v1_{stamp}.sha256'
checksum.write_text(''.join(lines))
print('DECISION:', decision)
print('CHECKSUM:', checksum)
PY
)
```

若 CSV 或某个候选的 metrics 不存在，先回第四步的汇总/诊断检查，不补造候选结果。

## 3. 进入 tmux 并检查 GPU

下面 smoke 和训练建议在 tmux 中运行。若已有合适会话，使用原会话即可。新建会话：

```bash
tmux new -s kcc-ac-transfer
```

如果提示会话已经存在：

```bash
tmux attach -t kcc-ac-transfer
```

在 tmux 中查看 GPU 状态：

```bash
nvidia-smi
```

下面命令沿用此前的物理 GPU 2。请把 `GPU=2` 改为管理员分配给你的 GPU；不能从本文判断 GPU 2 当前是否空闲。

脱离会话：按 `Ctrl+B`，松开，再按 `D`。之后用上面的 attach 命令返回。

## 4. 生成 MIT-States、C-GQA 配置

```bash
(
    set -euo pipefail
    source /data/xhw/miniconda3/etc/profile.d/conda.sh
    conda activate cpf
    ROOT=/data/xhw/czsl_repro
    PROJ="$ROOT/src/KCC0601_v1"
    EXP="$ROOT/experiments/kcc0601_glove_mvp_v1"
    CACHE_ROOT="$ROOT/knowledge/kcc0601_v1/cache_trusted_v1"

    cd "$PROJ"
    python tools/make_mvp_configs.py \
        --root "$ROOT" --exp "$EXP" --cache-root "$CACHE_ROOT" \
        --dataset mit-states --seed 0 --lambda-k 0.1
    python tools/make_mvp_configs.py \
        --root "$ROOT" --exp "$EXP" --cache-root "$CACHE_ROOT" \
        --dataset cgqa --seed 3351 --lambda-k 0.1

    python - "$EXP" <<'PY'
import sys, yaml
from pathlib import Path
exp = Path(sys.argv[1])
for ds, seed in [('mit-states', 0), ('cgqa', 3351)]:
    for variant in ['A', 'C']:
        path = exp/'configs'/f'mvp_{ds}_{variant}_legacy_s{seed}_l0.yml'
        cfg = yaml.safe_load(path.read_text())
        assert cfg['MODEL']['kcc_text'] == (variant == 'C'), path
        assert cfg['MODEL']['kcc_calib'] is False, path
        assert float(cfg['MODEL']['kcc_lambda']) == 0.0, path
        assert cfg['TRAIN']['kcc_test_at_end'] is False, path
        assert int(cfg['TRAIN']['max_epoch']) == 10, path
        print('A/C CONFIG PASSED:', path)
PY
)
```

现有工具会同时生成 A/C/D/E 配置。`--lambda-k 0.1` 只用于其中 D/E 的配置；本轮实际使用的 A/C 均为 lambda=0，且 calibration=False。不要运行生成的 D/E 配置。

首轮 MIT 用 seed=0，C-GQA 用 seed=3351；每个数据集内部的 A/C 使用相同 seed、10 epoch 配对预算。它是跨数据集工程验证，后续还需要多 seed。

若报告 `different config already exists`，保留现有配置并检查差异，不删除后强行覆盖。

## 5. 两数据集真实 batch smoke

此处使用已有 CPF baseline checkpoint 检查模型接入。smoke 内部包含 A/C/D/E 的数学及梯度检查；出现 D/E PASS 并不表示启动了 D/E 正式训练。

```bash
(
    set -euo pipefail
    source /data/xhw/miniconda3/etc/profile.d/conda.sh
    conda activate cpf
    ROOT=/data/xhw/czsl_repro
    PROJ="$ROOT/src/KCC0601_v1"
    EXP="$ROOT/experiments/kcc0601_glove_mvp_v1"
    CACHE_ROOT="$ROOT/knowledge/kcc0601_v1/cache_trusted_v1"
    AUDIT="$EXP/audit/ac_transfer_v1"
    GPU=2
    CK_M="$ROOT/runs/cpf_mit_states_seed0/checkpoints/mit-states-czsl_tttt/best_auc.pth"
    CK_G="$ROOT/runs/cpf_cgqa_seed3351_gpu2/checkpoints/cgqa-czsl_final_prediction/best_auc.pth"

    cd "$PROJ"
    test -s "$CK_M"
    test -s "$CK_G"
    mkdir -p "$AUDIT"
    for ds in mit-states cgqa; do
        if [ -e "$AUDIT/smoke_${ds}.json" ] || [ -e "$AUDIT/smoke_${ds}.log" ]; then
            printf 'Audit output exists; inspect it before reuse: %s\n' "$AUDIT"
            exit 2
        fi
    done

    CUDA_VISIBLE_DEVICES="$GPU" python -u tests/smoke_mvp.py \
        --cfg "$EXP/configs/mvp_mit-states_A_legacy_s0_l0.yml" \
        --checkpoint "$CK_M" --cache "$CACHE_ROOT/mit-states" \
        --out "$AUDIT/smoke_mit-states.json" --batch-size 2 \
        2>&1 | tee "$AUDIT/smoke_mit-states.log"

    CUDA_VISIBLE_DEVICES="$GPU" python -u tests/smoke_mvp.py \
        --cfg "$EXP/configs/mvp_cgqa_A_legacy_s3351_l0.yml" \
        --checkpoint "$CK_G" --cache "$CACHE_ROOT/cgqa" \
        --out "$AUDIT/smoke_cgqa.json" --batch-size 2 \
        2>&1 | tee "$AUDIT/smoke_cgqa.log"
)
```

两份 JSON 的 `status` 必须都是 `MVP_SMOKE_PASSED`。验证命令：

```bash
python - <<'PY'
import json
from pathlib import Path
audit = Path('/data/xhw/czsl_repro/experiments/kcc0601_glove_mvp_v1/audit/ac_transfer_v1')
for ds in ['mit-states', 'cgqa']:
    path = audit/f'smoke_{ds}.json'
    result = json.loads(path.read_text())
    if result.get('status') != 'MVP_SMOKE_PASSED':
        raise ValueError(path)
    print(ds, result['status'])
PY
```

C-GQA 的 OW 检查比 UT 大得多，需要等待加载和运算完成。如果 OOM，保留日志再处理实现与显存问题；当前 smoke 源码要求 batch-size>=2，不能直接改为 1 绕过。

如果中断或输出已存在，先检查 JSON 与日志。已通过且源码/cache/checkpoint 未变时无需重新 smoke；重新做检查则选新审计版本（例如 `ac_transfer_v2`），并同步修改训练门槛的 AUDIT 路径，不覆盖旧日志。

## 6. 通过 smoke 后训练四个 A/C run

在 tmux 中执行。已有旧 CPF checkpoint 用于 smoke；下面新 A/C 的配对训练仍需要完成，不能用旧 baseline 的数字替代本轮同预算 A。

```bash
(
    set -euo pipefail
    source /data/xhw/miniconda3/etc/profile.d/conda.sh
    conda activate cpf
    ROOT=/data/xhw/czsl_repro
    PROJ="$ROOT/src/KCC0601_v1"
    EXP="$ROOT/experiments/kcc0601_glove_mvp_v1"
    AUDIT="$EXP/audit/ac_transfer_v1"
    GPU=2
    cd "$PROJ"

    python - "$AUDIT" <<'PY'
import json, sys
from pathlib import Path
audit = Path(sys.argv[1])
for ds in ['mit-states', 'cgqa']:
    path = audit/f'smoke_{ds}.json'
    if json.loads(path.read_text()).get('status') != 'MVP_SMOKE_PASSED':
        raise ValueError(path)
print('MIT/C-GQA smoke gates passed')
PY

    for spec in "mit-states 0" "cgqa 3351"; do
        set -- $spec
        ds=$1
        seed=$2
        for variant in A C; do
            cfg="$EXP/configs/mvp_${ds}_${variant}_legacy_s${seed}_l0.yml"
            bash tools/run_mvp.sh "$cfg" "$GPU"
        done
    done
)
```

顺序为 MIT A → MIT C → C-GQA A → C-GQA C。同一块 GPU 顺序执行，不同时启动四个任务。每个任务包括训练及 val CW/OW 评估，不执行 test 评估。

目标目录分别为：

```text
experiments/kcc0601_glove_mvp_v1/runs/mvp_mit-states_A_legacy_s0_l0
experiments/kcc0601_glove_mvp_v1/runs/mvp_mit-states_C_legacy_s0_l0
experiments/kcc0601_glove_mvp_v1/runs/mvp_cgqa_A_legacy_s3351_l0
experiments/kcc0601_glove_mvp_v1/runs/mvp_cgqa_C_legacy_s3351_l0
```

每个 `status.txt` 应为 `train_and_val_complete`。`run_mvp.sh` 会拒绝覆盖已有 run。若中断，已完成的任务不用重跑；检查状态后只启动尚未完成的配置，失败目录另存/使用新版本处理，不能直接重复整段循环或删除证据。

## 7. 完成后汇总三数据集 A/C

仅当四个新 run 完成后执行：

```bash
python - <<'PY'
import csv, json
from pathlib import Path
exp = Path('/data/xhw/czsl_repro/experiments/kcc0601_glove_mvp_v1')
rows = []
for ds, seed in [('ut-zappos', 3351), ('mit-states', 0), ('cgqa', 3351)]:
    for setting in ['cw', 'ow']:
        values = {}
        for variant in ['A', 'C']:
            run = exp/'runs'/f'mvp_{ds}_{variant}_legacy_s{seed}_l0'
            if (run/'status.txt').read_text().strip() != 'train_and_val_complete':
                raise ValueError('Incomplete run: '+str(run))
            values[variant] = json.loads(
                (run/f'eval_val_{setting}'/'metrics.json').read_text())['metrics_raw']
        a, c = values['A'], values['C']
        row = {'dataset': ds, 'seed': seed, 'setting': setting}
        for name, key in [('AUC', 'AUC'), ('HM', 'best_hm'), ('unseen', 'best_unseen')]:
            row[f'A_{name}_pct'] = 100*a[key]
            row[f'C_{name}_pct'] = 100*c[key]
            row[f'delta_{name}_pp'] = 100*(c[key]-a[key])
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
dest = exp/'audit'/'ac_transfer_single_seed.csv'
if dest.exists():
    raise FileExistsError('Summary already exists; preserve it: '+str(dest))
dest.parent.mkdir(parents=True, exist_ok=True)
with dest.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('SUMMARY:', dest)
PY
```

先比较每个数据集内部 C-A 的 AUC/HM/unseen 差值。跨数据集首轮成功后再进入多 seed；单 seed 结果不足以报告显著性。若 C 只在 UT 有收益，先检查描述质量、OOV、聚合和归一化等原因，不使用 test 选择修改方案。

本步骤不会完成原设计中的所有模型变体。当前模型接入仅覆盖 A/C/D/E，其他变体仍需单独实现和验证。重构 K 时使用新 cache/protocol/run ID，不覆盖 v1 与旧负结果。
