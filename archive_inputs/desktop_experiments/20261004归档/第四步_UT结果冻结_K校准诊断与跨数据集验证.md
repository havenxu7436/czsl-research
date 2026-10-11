# 第四步：UT 结果冻结、K 校准诊断与跨数据集验证

## 0. 本步骤的起点与目标

UT-Zappos、seed=3351、legacy score、template K、λ=0.1 的 A/C/D/E 已完成。验证集结果为：

| setting | variant | AUC% | HM% | Seen% | Unseen% | ΔAUC pp | ΔHM pp |
|---|---|---:|---:|---:|---:|---:|---:|
| CW | A | 39.470 | 54.841 | 67.959 | 66.410 | 0 | 0 |
| CW | C | 44.262 | 57.894 | 68.871 | 71.502 | +4.791 | +3.053 |
| CW | D | 32.464 | 51.795 | 65.564 | 56.161 | -7.006 | -3.046 |
| CW | E | 37.239 | 53.897 | 66.135 | 64.741 | -2.231 | -0.944 |
| OW | A | 25.492 | 37.269 | 67.959 | 53.487 | 0 | 0 |
| OW | C | 32.079 | 46.367 | 68.871 | 59.435 | +6.587 | +9.097 |
| OW | D | 19.107 | 31.700 | 65.564 | 45.272 | -6.385 | -5.569 |
| OW | E | 23.972 | 36.262 | 66.135 | 53.830 | -1.520 | -1.008 |

当前结论：C 的知识文本 embedding 有明显收益，尤其改善 unseen；λ=0.1 的 K calibration 明显有害，E 被 calibration 拖累。CW AUC 交互效应约 -0.016 pp，接近线性叠加；OW HM 交互效应为 -4.536 pp，开放世界中存在额外冲突。

本步骤按以下顺序执行：

```text
冻结当前 A/C/D/E 结果
→ 在 A/C checkpoint 上做 validation-only 的 D-post/E-post λ 网格
→ 用预先声明的规则选择 λ 或拒绝当前 K
→ 审查 K 的 top/bottom 与 train/val 分布
→ 当前 K 无效时只扩展 A/C 到 MIT、C-GQA
→ 当前 K 有效时另立 calibration v2，不覆盖已有 D/E
```

本步骤不读取 test 指标，不在 test 上调 λ。所有旧 run、checkpoint、metrics 保留；不覆盖或删除。

---

## 1. 冻结现有 UT A/C/D/E

在服务器 Linux 终端执行：

```bash
(
    set -euo pipefail
    conda activate cpf

    ROOT=/data/xhw/czsl_repro
    EXP="$ROOT/experiments/kcc0601_glove_mvp_v1"
    PROTOCOL="$EXP/protocol"
    mkdir -p "$PROTOCOL"

    stamp=$(date +%Y%m%d_%H%M%S)
    manifest="$PROTOCOL/ut_acde_s3351_l0.1_${stamp}.sha256"

    for variant in A C D E; do
        lam=0
        if [ "$variant" = D ] || [ "$variant" = E ]; then
            lam=0.1
        fi

        run="$EXP/runs/mvp_ut-zappos_${variant}_legacy_s3351_l${lam}"
        cfg="$EXP/configs/mvp_ut-zappos_${variant}_legacy_s3351_l${lam}.yml"

        test -s "$run/status.txt"
        test "$(tr -d '\r\n' < "$run/status.txt")" = train_and_val_complete
        test -s "$run/eval_val_cw/metrics.json"
        test -s "$run/eval_val_ow/metrics.json"
        test -s "$cfg"

        ck=$(python - "$cfg" <<'PY'
import sys, yaml
c=yaml.safe_load(open(sys.argv[1]))
print(c['TRAIN']['checkpoint_dir']+'/'+c['config_name']+'_'+c['TRAIN']['train_name']+'/best_auc.pth')
PY
        )
        test -s "$ck"

        sha256sum "$cfg" "$run/status.txt" "$run/train.log" "$ck" \
            "$run/eval_val_cw/metrics.json" "$run/eval_val_ow/metrics.json" \
            >> "$manifest"
    done

    test -s "$manifest"
    echo "Frozen manifest: $manifest"
)
```

这一步只新增 checksum 清单，不改动实验结果。以后引用当前失败的 D/E 时，要写明 `template K, lambda=0.1, legacy, seed=3351`，不能用后续调参结果覆盖它。

---

## 2. 预先冻结 λ 诊断规则

本轮只使用已经声明过的候选：

```text
λ ∈ {0, 0.025, 0.05, 0.1, 0.2}
```

分别进行：

- D-post：使用 A checkpoint，只在推理时启用 K calibration。
- E-post：使用 C checkpoint，只在推理时启用 K calibration。

这是廉价诊断，不代替重新训练的 D/E。选择规则在运行前固定：

1. 主指标最大化 val CW AUC；
2. val CW HM 相对对应基线下降不超过 0.5 pp；
3. val OW AUC 相对对应基线下降不超过 0.5 pp；
4. val OW best-unseen 相对对应基线下降不超过 2 pp；
5. 若主指标相同，选择更小的 λ；
6. λ=0 必须复现 A/C；如果 λ=0 不能复现，停止并检查评估实现。

D-post 的对应基线是 A；E-post 的对应基线是 C。λ 的选择只用于诊断和后续 protocol revision，不能把调参后的 validation 数字当作独立确认集结论。

先把规则保存到服务器：

```bash
(
    set -euo pipefail
    ROOT=/data/xhw/czsl_repro
    EXP="$ROOT/experiments/kcc0601_glove_mvp_v1"
    mkdir -p "$EXP/protocol"

    rule="$EXP/protocol/lambda_post_selection_rule_v1.txt"
    test ! -e "$rule" || {
        echo "Rule file already exists: $rule"
        exit 2
    }

    printf '%s\n' \
      'dataset=ut-zappos' \
      'phase=val_only' \
      'candidates=0,0.025,0.05,0.1,0.2' \
      'primary=maximize_CW_AUC' \
      'constraint_CW_HM_delta_pp>=-0.5' \
      'constraint_OW_AUC_delta_pp>=-0.5' \
      'constraint_OW_unseen_delta_pp>=-2.0' \
      'tie_break=smaller_lambda' \
      'Dpost_baseline=A' \
      'Epost_baseline=C' \
      > "$rule"

    sha256sum "$rule"
)
```

---

## 3. 在 A/C checkpoint 上运行 D-post/E-post λ 网格

确认没有训练进程占用目标 GPU：

```bash
nvidia-smi
ps -eo pid,etime,cmd | grep '[p]ython.*train.py' || true
```

选择空闲物理卡。下面示例使用 GPU 2；若实际空闲卡不同，只修改 `GPU=2`。

```bash
(
    ROOT=/data/xhw/czsl_repro
    PROJ="$ROOT/src/KCC0601_v1"
    EXP="$ROOT/experiments/kcc0601_glove_mvp_v1"
    DIAG="$EXP/diagnostics/lambda_post_v1"
    GPU=3

    cd "$PROJ"
    mkdir -p "$DIAG"
    export CUDA_VISIBLE_DEVICES="$GPU"

    for base in A C; do
        if [ "$base" = A ]; then
            label=Dpost
        else
            label=Epost
        fi

        cfg="$EXP/configs/mvp_ut-zappos_${base}_legacy_s3351_l0.yml"
        test -s "$cfg"

        ck=$(python - "$cfg" <<'PY'
import sys, yaml
c=yaml.safe_load(open(sys.argv[1]))
print(c['TRAIN']['checkpoint_dir']+'/'+c['config_name']+'_'+c['TRAIN']['train_name']+'/best_auc.pth')
PY
        )
        test -s "$ck"

        for lam in 0.0 0.025 0.05 0.1 0.2; do
            for setting in cw ow; do
                ow=False
                if [ "$setting" = ow ]; then ow=True; fi

                tag="${label}_l${lam}_${setting}"
                out="$DIAG/$tag"
                log="$DIAG/${tag}.log"

                if [ -s "$out/metrics.json" ]; then
                    echo "Already complete, skip: $out"
                    continue
                fi
                test ! -e "$out" || {
                    echo "Partial output exists; inspect it and use a new DIAG version: $out"
                    exit 2
                }

                python tools/eval_mvp.py \
                    --cfg "$cfg" --load "$ck" --phase val \
                    --setting "$setting" --out "$out" \
                    EVAL.open_world "$ow" \
                    MODEL.kcc_calib True \
                    MODEL.kcc_lambda "$lam" \
                    TRAIN.kcc_variant "$label" \
                    TRAIN.kcc_stage tune_lambda_post \
                    2>&1 | tee "$log"
            done
        done
    done
)
```

该命令共进行20次 UT validation evaluation，不重新训练，不访问 test。每次重新初始化模型并读取 GloVe，因此总时间会比单次评估长。若 SSH 不稳定，在 tmux 中运行：

```bash
tmux new -s kcc-lambda-post
```

离开 tmux：`Ctrl+B` 后按 `D`；重新连接：

```bash
tmux attach -t kcc-lambda-post
```

---

## 4. 汇总 λ 网格并按冻结规则选择

```bash
ROOT=/data/xhw/czsl_repro
EXP="$ROOT/experiments/kcc0601_glove_mvp_v1"
DIAG="$EXP/diagnostics/lambda_post_v1"

python - "$EXP" "$DIAG" <<'PY'
import csv, json, sys
from pathlib import Path

exp, diag = map(Path, sys.argv[1:])
lambdas = [0.0, 0.025, 0.05, 0.1, 0.2]
definitions = {
    'Dpost': ('A', 'mvp_ut-zappos_A_legacy_s3351_l0'),
    'Epost': ('C', 'mvp_ut-zappos_C_legacy_s3351_l0'),
}

def read(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())['metrics_raw']

rows=[]
selection={}
for label,(baseline_name,run_id) in definitions.items():
    base={s:read(exp/'runs'/run_id/f'eval_val_{s}'/'metrics.json')
          for s in ('cw','ow')}
    candidates=[]
    for lam in lambdas:
        got={s:read(diag/f'{label}_l{lam}_{s}'/'metrics.json')
             for s in ('cw','ow')}
        row={
            'diagnostic':label,
            'baseline':baseline_name,
            'lambda':lam,
            'cw_auc_pct':100*got['cw']['AUC'],
            'cw_hm_pct':100*got['cw']['best_hm'],
            'ow_auc_pct':100*got['ow']['AUC'],
            'ow_hm_pct':100*got['ow']['best_hm'],
            'ow_unseen_pct':100*got['ow']['best_unseen'],
            'delta_cw_auc_pp':100*(got['cw']['AUC']-base['cw']['AUC']),
            'delta_cw_hm_pp':100*(got['cw']['best_hm']-base['cw']['best_hm']),
            'delta_ow_auc_pp':100*(got['ow']['AUC']-base['ow']['AUC']),
            'delta_ow_hm_pp':100*(got['ow']['best_hm']-base['ow']['best_hm']),
            'delta_ow_unseen_pp':100*(got['ow']['best_unseen']-base['ow']['best_unseen']),
        }
        row['eligible']=(row['delta_cw_hm_pp'] >= -0.5 and
                         row['delta_ow_auc_pp'] >= -0.5 and
                         row['delta_ow_unseen_pp'] >= -2.0)
        rows.append(row)
        if row['eligible']:
            candidates.append(row)

    zero=next(r for r in rows if r['diagnostic']==label and r['lambda']==0.0)
    for key in ('delta_cw_auc_pp','delta_cw_hm_pp','delta_ow_auc_pp',
                'delta_ow_hm_pp','delta_ow_unseen_pp'):
        if abs(zero[key]) > 1e-6:
            raise AssertionError(f'{label} lambda=0 identity failed: {key}={zero[key]}')

    if not candidates:
        raise AssertionError(label+' has no eligible candidate; lambda=0 should be eligible')
    chosen=max(candidates,key=lambda r:(r['cw_auc_pct'],-r['lambda']))
    selection[label]={
        'selected_lambda':chosen['lambda'],
        'selection_metric':'val_cw_auc',
        'selected_row':chosen,
        'positive_lambda_selected':chosen['lambda'] > 0,
    }

dest=diag/'lambda_post_results.csv'
with dest.open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
(diag/'lambda_selection.json').write_text(
    json.dumps(selection,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

for row in rows:
    print(json.dumps(row,ensure_ascii=False))
print('\nSELECTION')
print(json.dumps(selection,ensure_ascii=False,indent=2))
print('\nCSV:',dest)
print('JSON:',diag/'lambda_selection.json')
PY
```

再冻结诊断结果：

```bash
ROOT=/data/xhw/czsl_repro
EXP="$ROOT/experiments/kcc0601_glove_mvp_v1"
DIAG="$EXP/diagnostics/lambda_post_v1"
stamp=$(date +%Y%m%d_%H%M%S)

find "$DIAG" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$EXP/protocol/lambda_post_v1_${stamp}.sha256"
```

---

## 5. 审查当前 K，但不使用 test 调参

### 5.1 查看 top/bottom pair 与原始分布

```bash
ROOT=/data/xhw/czsl_repro
CACHE="$ROOT/knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k"

python - "$CACHE" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
r=json.loads((p/'report.json').read_text(encoding='utf-8'))
for key in ('raw_template_K','K_oa_text','centered_prior',
            'description_OOV_fraction','description_fallback_sentences',
            'top_bottom_pairs'):
    print('\n'+key)
    print(json.dumps(r[key],ensure_ascii=False,indent=2))
PY
```

人工检查 top pair 是否只是名称/描述词形重合，bottom pair 是否包含视觉上合理组合。发现明显反例时记录，不直接改原 cache。

### 5.2 比较 train/val pair 的 prior 分布

下面只使用 train 与 val；不读取 test pair 列表。

```bash
(
    set -euo pipefail
    ROOT=/data/xhw/czsl_repro
    CACHE="$ROOT/knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k"
    DATA="$ROOT/data/ut-zap50k/compositional-split-natural"

    python - "$CACHE" "$DATA" <<'PY'
import json,sys,torch
from pathlib import Path
cache,data=map(Path,sys.argv[1:])
labels=json.loads((cache/'labels.json').read_text())
attrs,objs=labels['attrs'],labels['objs']
ai={x:i for i,x in enumerate(attrs)}
oi={x:i for i,x in enumerate(objs)}
k=torch.load(cache/'K_oa_prior_centered.pt',map_location='cpu').float()

def pairs(name):
    out=[]
    for line in (data/(name+'_pairs.txt')).read_text().splitlines():
        if line.strip():
            a,o=line.split()
            out.append((a,o))
    return out

def report(name,items):
    v=torch.tensor([k[oi[o],ai[a]].item() for a,o in items])
    print(name,{
        'count':len(items),
        'mean':v.mean().item(),
        'population_std':v.std(unbiased=False).item(),
        'min':v.min().item(),
        'median':v.median().item(),
        'max':v.max().item(),
    })

report('train',pairs('train'))
report('val',pairs('val'))
print('full_cartesian',{
    'count':k.numel(),
    'mean':k.mean().item(),
    'population_std':k.std(unbiased=False).item(),
    'min':k.min().item(),
    'median':k.median().item(),
    'max':k.max().item(),
})
PY
)
```

如果 train pair 的 prior 明显高于 val pair，K 可能在强化 seen-composition bias；这可以解释 D 的 unseen 大幅下降。该检查只用于诊断，不能据此手工编辑 val pair 的 K。

---

## 6. 根据 λ 结果选择后续分支

读取选择结果：

```bash
ROOT=/data/xhw/czsl_repro
EXP="$ROOT/experiments/kcc0601_glove_mvp_v1"
python -m json.tool "$EXP/diagnostics/lambda_post_v1/lambda_selection.json"
```

### 分支 A：D-post 和 E-post 都选择 λ=0

结论：当前 template K 在已声明网格中没有校准价值。执行：

1. 保留 λ=0.1 的 D/E 作为负结果；
2. 停止用当前 K 训练更多 D/E；
3. 不在 MIT/C-GQA 上复制有害的 λ=0.1 calibration；
4. 先把成功的 C 路径扩展到 MIT/C-GQA；
5. 后续若重构 K，建立 `cache_trusted_v2` 和新 protocol，不覆盖 `cache_trusted_v1`。

### 分支 B：至少一个诊断选择 λ>0

结论：K 可能有信息，但 λ=0.1 或训练/推理尺度不合适。执行：

1. 将 D-post/E-post 仅标为 validation diagnostic；
2. 不把它写成重新训练的 D/E；
3. 建立 `legacy_calib_v2` 协议；
4. 修正训练与推理的 calibration 尺度后更新 smoke；
5. 使用新的 run ID 重新训练 D-v2/E-v2，旧 D/E 保留；
6. D-v2/E-v2 通过 UT 后，才扩展到 MIT/C-GQA。

当前实现训练时 composition cosine 被 `1/0.05=20` 放大，而验证时最终分数由未放大的 cosine 相加；同一 λ 在两个阶段的相对强度不同。`legacy_calib_v2` 至少要预先固定以下一种定义：

```text
方案1：calibration只作为post-hoc评分项，D-post/E-post独立命名；
方案2：把prior定义成composition cosine residual，训练时按1/temp同步缩放；
方案3：改成统一条件概率logit协议，A-prob/D-prob/E-prob全部重跑。
```

不要在得到 λ 网格结果后直接修改旧 D/E 目录。任何尺度修订都要更新源码 hash、smoke 和 run ID。

---

## 7. 扩展成功的 A/C 到 MIT-States 与 C-GQA

这一部分不依赖 calibration 是否成功。建议先完成第3–6节，再顺序运行。

### 7.1 生成配置

```bash
(
    set -euo pipefail
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
)
```

工具会同时生成 D/E 配置，但本阶段只执行 A/C。

### 7.2 两数据集真实 batch smoke

```bash
(
    set -euo pipefail
    conda activate cpf
    ROOT=/data/xhw/czsl_repro
    PROJ="$ROOT/src/KCC0601_v1"
    EXP="$ROOT/experiments/kcc0601_glove_mvp_v1"
    CACHE_ROOT="$ROOT/knowledge/kcc0601_v1/cache_trusted_v1"
    GPU=2

    CK_M="$ROOT/runs/cpf_mit_states_seed0/checkpoints/mit-states-czsl_tttt/best_auc.pth"
    CK_G="$ROOT/runs/cpf_cgqa_seed3351_gpu2/checkpoints/cgqa-czsl_final_prediction/best_auc.pth"

    cd "$PROJ"
    test -s "$CK_M"
    test -s "$CK_G"

    CUDA_VISIBLE_DEVICES="$GPU" python tests/smoke_mvp.py \
        --cfg "$EXP/configs/mvp_mit-states_A_legacy_s0_l0.yml" \
        --checkpoint "$CK_M" --cache "$CACHE_ROOT/mit-states" \
        --out "$EXP/audit/smoke_mit-states_s0.json" --batch-size 2 \
        2>&1 | tee "$EXP/audit/smoke_mit-states_s0.log"

    CUDA_VISIBLE_DEVICES="$GPU" python tests/smoke_mvp.py \
        --cfg "$EXP/configs/mvp_cgqa_A_legacy_s3351_l0.yml" \
        --checkpoint "$CK_G" --cache "$CACHE_ROOT/cgqa" \
        --out "$EXP/audit/smoke_cgqa_s3351.json" --batch-size 2 \
        2>&1 | tee "$EXP/audit/smoke_cgqa_s3351.log"
)
```

两份 JSON 都必须包含：

```text
MVP_SMOKE_PASSED
```

C-GQA 的 OW candidate 数量大，smoke 时间和显存会明显高于 UT。若发生 OOM，保留日志，不降低数学定义；先把 `--batch-size 2` 改为新的 smoke 版本支持 batch1/冻结BN，或增加分块实现后再验收。当前脚本要求 batch-size≥2，不要直接传1绕过。

### 7.3 只训练 A/C

建议在 tmux 中执行：

```bash
tmux new -s kcc-ac-transfer
```

```bash
(
    set -euo pipefail
    conda activate cpf
    ROOT=/data/xhw/czsl_repro
    PROJ="$ROOT/src/KCC0601_v1"
    EXP="$ROOT/experiments/kcc0601_glove_mvp_v1"
    GPU=2

    cd "$PROJ"

    python - "$EXP/audit/smoke_mit-states_s0.json" "$EXP/audit/smoke_cgqa_s3351.json" <<'PY'
import json,sys
for path in sys.argv[1:]:
    r=json.load(open(path))
    assert r['status']=='MVP_SMOKE_PASSED', path
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

每个 run 的 `status.txt` 必须为 `train_and_val_complete`。如已有同名目录，脚本会拒绝覆盖；先判断它是完成还是失败，不能删除证据后重跑。

---

## 8. 汇总三数据集 A/C 单 seed 迁移结果

```bash
ROOT=/data/xhw/czsl_repro
EXP="$ROOT/experiments/kcc0601_glove_mvp_v1"

python - "$EXP" <<'PY'
import csv,json,sys
from pathlib import Path
exp=Path(sys.argv[1])
datasets={
    'ut-zappos':3351,
    'mit-states':0,
    'cgqa':3351,
}
rows=[]
for ds,seed in datasets.items():
    for setting in ('cw','ow'):
        values={}
        for variant in ('A','C'):
            run=f'mvp_{ds}_{variant}_legacy_s{seed}_l0'
            path=exp/'runs'/run/f'eval_val_{setting}'/'metrics.json'
            if not path.is_file(): raise FileNotFoundError(path)
            values[variant]=json.loads(path.read_text())['metrics_raw']
        a,c=values['A'],values['C']
        row={
            'dataset':ds,'seed':seed,'setting':setting,
            'A_AUC_pct':100*a['AUC'],'C_AUC_pct':100*c['AUC'],
            'delta_AUC_pp':100*(c['AUC']-a['AUC']),
            'A_HM_pct':100*a['best_hm'],'C_HM_pct':100*c['best_hm'],
            'delta_HM_pp':100*(c['best_hm']-a['best_hm']),
            'A_unseen_pct':100*a['best_unseen'],
            'C_unseen_pct':100*c['best_unseen'],
            'delta_unseen_pp':100*(c['best_unseen']-a['best_unseen']),
        }
        rows.append(row)
        print(json.dumps(row,ensure_ascii=False))

dest=exp/'audit/ac_transfer_single_seed.csv'
with dest.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]))
    w.writeheader();w.writerows(rows)
print('CSV:',dest)
PY
```

判断：

- C 在至少2/3数据集的主指标上优于 A，并且 OW 没有明显崩溃：进入 A/C 多 seed 确认；
- C 只在 UT 有效：检查不同数据集描述 OOV、知识质量、名称/描述聚合和归一化，不直接扩张完整矩阵；
- C 在 MIT/C-GQA 反向：保留负结果，检查是否是描述噪声或 embedding 尺度，不用 test 选方案。

单 seed 结果只能作为工程 Gate，不能作为正式统计结论。

---

## 9. 多 seed 与后续模型的进入条件

当 A/C 跨数据集 Gate 通过后，冻结：

```text
cache manifest
源码 hash
Text K 聚合规则
10 epochs / batch64 / optimizer与学习率
validation-best CW AUC checkpoint规则
CW/OW candidate定义
seeds={0,3351,2026}
```

然后为三数据集分别运行 A/C 的相同三 seed，报告：

- 每个 seed 的 A、C 绝对值；
- 每个 seed 的配对差 `C−A`；
- 配对差均值和标准差；
- CW AUC/HM、OW AUC/HM、best seen/unseen；
- 失败 seed 及原因，不能只保留提升的 seed。

在以下条件满足前，不开始 B/F/G/H：

1. A/C 的跨数据集 smoke 与单 seed Gate 完成；
2. λ 诊断已经明确接受或拒绝当前 K；
3. 若保留 calibration，已完成 v2 尺度协议和新版 smoke；
4. Logic adapter 与 image-conditioned feature K 有各自的梯度、关闭恒等和候选顺序测试。

---

## 10. 本步骤结束时应保留的文件

```text
$EXP/protocol/
├── ut_acde_s3351_l0.1_<timestamp>.sha256
├── lambda_post_selection_rule_v1.txt
└── lambda_post_v1_<timestamp>.sha256

$EXP/diagnostics/lambda_post_v1/
├── Dpost_l{lambda}_{cw,ow}/metrics.json
├── Epost_l{lambda}_{cw,ow}/metrics.json
├── lambda_post_results.csv
└── lambda_selection.json

$EXP/audit/
├── smoke_mit-states_s0.json
├── smoke_cgqa_s3351.json
└── ac_transfer_single_seed.csv
```

第四步的核心结果不是“强行找到非零 λ”，而是可审计地回答两个问题：当前 K 在合理权重下是否有校准价值，以及 C 的文本知识增益能否跨数据集复现。
