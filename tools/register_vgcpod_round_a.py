"""Import experiment 02, verify its files, and restore the original A training package."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'VG-CPOD方案/02'
PACKAGE = 'vgcpod_v1_roundA'
TITLE = '支持损失重加权与 CP 分组学习率'
OLD = ROOT / 'VG-CPOD方案/v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def table(headers, values):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in values])


def import_data(source):
    source = Path(source).resolve()
    if STAGE.exists():
        raise FileExistsError(STAGE)
    STAGE.mkdir(parents=True)
    package_map = {}
    package = source / PACKAGE
    for p in sorted(package.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix == '.pyc':
            continue
        rel = p.relative_to(package).as_posix()
        if rel.startswith('configs/'):
            dest = STAGE / rel
        elif rel.startswith(('assets/', 'references/')):
            dest = STAGE / 'configs' / rel
        elif rel.startswith('validation/') or rel == 'release_manifest.json':
            dest = STAGE / 'results/package_validation' / rel
        elif p.suffix in ('.md', '.html'):
            dest = STAGE / 'results/package_support' / (rel + '.txt')
        else:
            dest = STAGE / 'code' / PACKAGE / rel
        copy(p, dest)
        package_map[rel] = dest.relative_to(STAGE).as_posix()
    write(STAGE / 'package_members.json', {'package': PACKAGE, 'members': package_map})
    for name in ['vgcpod_v1_roundA_20261019.zip', 'vgcpod_v1_roundA_20261019.zip.sha256',
                 'vgcpod_roundA_confirm_20261019.tar.gz']:
        copy(source / name, STAGE / 'results/archives' / name)
    arc = STAGE / 'results/archives/vgcpod_roundA_confirm_20261019.tar.gz'
    arc.with_name(arc.name + '.sha256').write_text(sha(arc) + '  ' + arc.name + '\n', encoding='ascii')
    raw = source / 'server_results_roundA_confirm_20261019/roundA_20261019'
    for p in raw.rglob('*'):
        if p.is_file() and p.name != '.round_a.lock':
            copy(p, STAGE / 'results/raw_export/roundA_20261019' / p.relative_to(raw))
    # Keep the first launcher state without duplicating the six unchanged seed3351 models.
    first = source / 'server_results_roundA_20261019/roundA_20261019'
    for p in first.iterdir():
        if p.is_file() and p.name != '.round_a.lock':
            copy(p, STAGE / 'results/initial_launch' / p.name)
    for folder, target in [('analysis_roundA_20261019', 'analysis/single_seed'),
                           ('analysis_roundA_confirm_20261019', 'analysis/confirmed')]:
        for p in (source / folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.csv', '.json', '.png', '.txt'):
                copy(p, STAGE / target / p.relative_to(source / folder))
    for folder in ['analysis_roundA_20261019', 'analysis_roundA_confirm_20261019']:
        for p in (source / folder).glob('*.py'):
            copy(p, STAGE / 'code/original_analysis' / folder / p.name)
    copy(source / 'vgcpod_v1_roundA/validation/source_changes.patch', STAGE / 'code/source_changes.patch')
    for p in source.glob('VG-CPOD_v1_A组*.md'):
        copy(p, STAGE / 'results/package_support/original_reports' / (p.name + '.txt'))
    create_recompute(source)
    generate_readmes()
    inventory = []
    for p in sorted((STAGE / 'results/raw_export/roundA_20261019').glob('A*/ut-zap50k/V*/seed*/metrics.json')):
        m, protocol = read(p), read(p.parent / 'protocol.json')
        inventory.append({'case': protocol['config']['experiment_id'], 'seed': protocol['seed'],
                          'selected_epoch': m['selected_epoch'], 'path': p.parent.relative_to(STAGE).as_posix(),
                          'config': protocol['config'], 'metrics': m['metrics']})
    write(STAGE / 'configs/experiment_inventory.json', inventory)
    write(STAGE / 'runbook.json', {'line': 'VG-CPOD方案', 'stage': '02', 'primary_package': PACKAGE,
        'commands': [{'source': 'README.md', 'language': lang, 'text': text}
                     for lang, text in re.findall(r'```(bash|powershell)\n(.*?)```', (STAGE / 'README.md').read_text(encoding='utf-8'), re.S)]})
    write(STAGE / 'external_inputs.json', {'framework': 'frozen CLIP ViT-L/14',
        'cache': '/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14',
        'feature_manifest_sha256': 'bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a',
        'feature_arrays': 'Existing server cache; images and CLIP weights are needed only to rebuild it.',
        'original_input_index': '../v1/external_inputs.json'})
    write(STAGE / 'inputs.json', {'line': 'VG-CPOD方案', 'stage': '02', 'version': 'v1',
        'title': TITLE, 'knowledge': [], 'completed_runs': 12, 'evaluations': 48,
        'confirmed_cases': ['A1', 'A5', 'A6'], 'confirmed_seeds': [3351, 3352, 3353],
        'initial_only_cases': ['A2', 'A3', 'A4'], 'initial_seed': 3351,
        'package_member_map': 'package_members.json', 'experiment_inventory': 'configs/experiment_inventory.json',
        'code_files': [p.relative_to(STAGE).as_posix() for p in sorted((STAGE / 'code').rglob('*')) if p.is_file()],
        'configs': [p.relative_to(STAGE).as_posix() for p in sorted((STAGE / 'configs').rglob('*.json'))],
        'documents': [p.relative_to(STAGE).as_posix() for p in sorted(STAGE.rglob('README.md'))]})
    finalize()


def create_recompute(source):
    original = (source / 'analysis_roundA_confirm_20261019/analyze_confirm.py').read_text(encoding='utf-8')
    start = original.index('ROOT=Path(')
    end = original.index('SEEDS=', start)
    original = original[:start] + """STAGE=Path(__file__).resolve().parent.parent
ROOT=STAGE.parent.parent
OUT=Path(__file__).resolve().parent/'recomputed'
RUNS=STAGE/'results/raw_export/roundA_20261019'
OLD=STAGE.parent/'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs'
PKG=STAGE/'code/vgcpod_v1_roundA'
""" + original[end:]
    original = original.replace("ROOT/'analysis_20261018/analyze_results.py'", "STAGE.parent/'v1/code/analysis_20261018/analyze_results.py'")
    original = original.replace("ROOT/'vgcpod_roundA_confirm_20261019.tar.gz'", "STAGE/'results/archives/vgcpod_roundA_confirm_20261019.tar.gz'")
    begin = original.index("    for c in ['A1','A2','A3','A4','A5','A6']:")
    end = original.index('    errors=dict(', begin)
    original = original[:begin] + """    # Recheck the original 78 seed3351 hashes against the original first-run inventory.
    first_hashes=read(STAGE/'results/initial_seed3351_hashes.json')
    for rel, fingerprint in first_hashes.items():
        assert sha(RUNS/rel)==fingerprint,rel
        audit['unchanged_seed3351_files']+=1
""" + original[end:]
    original = original.replace("read(PKG/'configs/round_a'/(c+'.json'))", "read(STAGE/'configs/round_a'/(c+'.json'))")
    original = original.replace("for name,fingerprint in p['source_sha256'].items():assert sha(PKG/name)==fingerprint",
        "for name,fingerprint in p['source_sha256'].items():assert sha(STAGE/read(STAGE/'package_members.json')['members'][name])==fingerprint")
    original = '# Repository paths adapted from the original analysis; numerical calculations unchanged.\n' + original
    (STAGE / 'analysis/recompute.py').write_text(original, encoding='utf-8')
    first = source / 'server_results_roundA_20261019/roundA_20261019'
    hashes = {p.relative_to(first).as_posix(): sha(p) for p in sorted(first.glob('A*/ut-zap50k/V*/seed3351/**/*')) if p.is_file()}
    write(STAGE / 'results/initial_seed3351_hashes.json', hashes)
    (STAGE / '.gitignore').write_text('analysis/recomputed/\n', encoding='utf-8')


def generate_readmes():
    data = read(STAGE / 'analysis/confirmed/analysis.json')
    agg = {(r['case'], r['setting']): r for r in data['aggregate']}
    diag = {(r['case'], r['phase']): r for r in data['diagnostics']}
    fmt = lambda r, k: f"{r[k+'_mean']:.3f} ± {r[k+'_sd']:.3f}"
    main_table = table(['组', 'VAL CW AUC', 'VAL OW AUC', 'TEST CW AUC', 'TEST OW AUC'],
        [[c] + [fmt(agg[c, s], 'AUC_native') for s in ['val_cw', 'val_ow', 'test_cw', 'test_ow']]
         for c in ['D0', 'V1', 'A1', 'A5', 'A6']])
    diag_table = table(['组', '已见条件属性准确率', '未见条件属性准确率', '未见图像的已见概率质量', '物体准确率'],
        [[c] + [fmt(diag[c, 'test'], k) for k in ['attr_seen', 'attr_unseen', 'support_unseen', 'object_acc']]
         for c in ['D0', 'V1', 'A1', 'A5', 'A6']])
    seed_rows = []
    raw = STAGE / 'results/raw_export/roundA_20261019'
    for p in sorted(raw.glob('A*/ut-zap50k/V*/seed*/metrics.json')):
        m, pr = read(p), read(p.parent / 'protocol.json')
        seed_rows.append([pr['config']['experiment_id'], pr['seed'], m['selected_epoch']] +
            [f"{m['metrics'][s]['AUC_native']:.4f}" for s in ['val_cw', 'val_ow', 'test_cw', 'test_ow']])
    seed_table = table(['组', 'seed', '最佳 epoch', 'VAL CW', 'VAL OW', 'TEST CW', 'TEST OW'], seed_rows)
    pp = []
    for comp in ['A1-V1', 'A5-V1', 'A5-D0', 'A5-A1', 'A5-A6']:
        q = [next(r for r in data['paired'] if r['comparison'] == comp and r['setting'] == s and r['metric'] == 'AUC_native')
             for s in ['test_cw', 'test_ow']]
        pp.append([comp] + [f"{r['mean']:+.3f} ± {r['sd']:.3f}" for r in q] + [f"{r['positive']}/3" for r in q])
    paired = table(['配对比较', 'TEST CW 增量', 'TEST OW 增量', 'CW 提高 seed', 'OW 提高 seed'], pp)
    configs = []
    for c in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']:
        cfg = read(STAGE / 'configs/round_a' / (c + '.json'))
        cp = '关闭' if c == 'A6' else f"{cfg['lr']*cfg['cp_lr_scale']:.0e} → {cfg['min_lr']*cfg['cp_lr_scale']:.0e}"
        configs.append([f'[{c}](configs/round_a/{c}.json)', cfg['variant'], cp,
                        cfg['task_aux_beta'], '3351/3352/3353' if c in ['A1', 'A5', 'A6'] else '3351'])
    matrix = table(['组', '模型', 'CP 学习率起点→终点', 'β', '已完成 seed'], configs)
    main = r'''# 02 · 支持损失重加权与 CP 分组学习率

**A 组实验，VG-CPOD v1，2026-10-19。** 主框架为冻结 CLIP ViT-L/14。UT-Zappos natural split，共完成 12 次训练、120 个模型 epoch、48 份 VAL/TEST CW/OW 评价；A1/A5/A6 各三个 seed，A2/A3/A4 各一个 seed。

[代码与具体改动](code/README.md) · [超参数配置](configs/README.md) · [全部训练结果](results/README.md) · [数据分析](analysis/README.md) · [01 原实验](../v1/README.md)

## 1. 实验目的与修订范围

原静态 CP 在 CW 上优于 D0，但 OW 与未见条件属性识别下降。本轮分别降低 CP 学习率、降低训练组合支持损失权重，再检验两项调整的联合效果。

图文编码、缓存、全局静态 CP 前向及概率链式分解沿用 01/V1。新增修订集中在训练目标、优化器学习率分组、诊断输出，以及三 GPU 与 tmux 队列。A6 为相同 β 下的 V0/D0 对照。

## 2. 公式修订：分离组合判别与训练支持压力

全局静态 CP 修正保持为：

```math
u_i=A_u e_i^a,\quad v_j=A_v e_j^o,\quad
\Delta U_{bji}=\rho\kappa\,g_b^\top W_c(u_i\odot v_j),\quad
U_{bji}=U^{\mathrm{base}}_{bji}+\Delta U_{bji}.
```

联合概率在全部属性与物体的笛卡尔积上归一化，训练时的组合交叉熵在训练组合内计算：

```math
\begin{aligned}
p_{ij}(x)&=p(o_j\mid x)\,p(a_i\mid o_j,x),\qquad \sum_j\sum_i p_{ij}(x)=1,\\
Z_s(x)&=\sum_{(i,j)\in\mathcal{C}^{s}}p_{ij}(x),\\
\ell_{\mathrm{pair}}&=-\log\frac{p_{a o}(x)}{Z_s(x)},\qquad
\ell_{\mathrm{support}}=-\log Z_s(x),\\
\ell_{\mathrm{cond}}+\ell_{\mathrm{obj}}&=-\log p_{a o}(x)
=\ell_{\mathrm{pair}}+\ell_{\mathrm{support}}.
\end{aligned}
```

这里 $a,o$ 是该训练样本的真实属性与物体。原目标可精确改写为：

```math
\ell_{\mathrm{old}}
=\ell_{\mathrm{pair}}+0.5\ell_{\mathrm{cond}}+0.5\ell_{\mathrm{obj}}
=1.5\ell_{\mathrm{pair}}+0.5\ell_{\mathrm{support}}.
```

新增 β 归一化目标：

```math
\begin{aligned}
\ell_{\mathrm{task}}(\beta)
&=\frac{1.5}{1+\beta}\left[\ell_{\mathrm{pair}}
+\beta(\ell_{\mathrm{cond}}+\ell_{\mathrm{obj}})\right]\\
&=1.5\ell_{\mathrm{pair}}+\frac{1.5\beta}{1+\beta}\ell_{\mathrm{support}}.
\end{aligned}
```

| β | 组合判别项系数 | 支持损失系数 | 相对原目标 |
|---|---|---|---|
| 0.5 | 1.5 | 0.5 | 恢复原目标 |
| 0.25 | 1.5 | 0.3 | 支持项减少 40% |
| 0.1 | 1.5 | 0.13636 | 支持项减少 72.73% |

归一化保持组合判别项系数为 1.5，使 β 的比较集中于支持损失。上述恒等式对每个样本成立，取 batch 均值后同样成立。代码在 β=0.5 时保留原来的浮点运算顺序。

## 3. 公式修订：CP 使用固定比例的分组学习率

适配器参数组为 `Bo/Co/Bc/Cc`，静态 CP 参数组为 `Au/Av/Wc`。两组使用同一余弦进程：

```math
\begin{aligned}
\eta_t&=\eta_{\min}+\frac{\eta_{\max}-\eta_{\min}}{2}
\left[1+\cos\left(\frac{\pi t}{T-1}\right)\right],\quad t=0,\ldots,T-1,\\
\eta_t^{\mathrm{adapter}}&=\eta_t,\qquad
\eta_t^{\mathrm{CP}}=s_{\mathrm{CP}}\eta_t,\qquad
s_{\mathrm{CP}}\in\{1,0.3,0.1\}.
\end{aligned}
```

训练图像 22,998 张，batch64，每轮 360 次更新，10 轮共 $T=3600$。适配器保持 1e-3→1e-5；CP 比较 1e-3→1e-5、3e-4→3e-6、1e-4→1e-6。

AdamW 的矩估计、weight decay 与总梯度裁剪沿用原实现。分组学习率缩放 AdamW 更新步及其解耦权重衰减步；梯度裁剪仍在所有可训练参数上统一计算。每一步都乘 `lr_scale`，从而保持组间比例。

## 4. 具体代码修订

| 文件 | 已实施的修改 | 实验作用 |
|---|---|---|
| [model.py](code/vgcpod_v1_roundA/vgcpod/model.py) | `task_aux_beta` 控制归一化目标；新增 `task/support` | 直接调节支持损失，记录等价分解 |
| [engine.py](code/vgcpod_v1_roundA/vgcpod/engine.py) | `make_optimizer` 分组；`set_learning_rates` 每步保留 `lr_scale` | CP 与适配器使用不同更新尺度 |
| 同上 | 同一 VAL 前向计算 CW/OW；新增支持概率质量、未见条件属性、中心化修正 RMS、实际裁剪步骤比例 | 解释性能变化来自哪里 |
| [run.py](code/vgcpod_v1_roundA/run.py) | `--variant` 省略时采用配置文件值 | A6 按配置正确运行 V0 |
| [round_a.py](code/vgcpod_v1_roundA/round_a.py) | 三卡队列、同一实验续训、完成任务跳过、逐运行汇总 | 运行初始六组及后续补 seed |
| [scripts](code/vgcpod_v1_roundA/scripts/) | tmux 启动、记录退出码、按 GPU 分配独立模型 | 三 GPU 持续运行 |
| [tests](code/vgcpod_v1_roundA/tests/) | 原数学检查与 A 组集成检查共 17 项 | 保存已交付版本的检查记录 |

核心损失实现：

```python
beta = self.cfg.get('task_aux_beta', 0.5)
task = (pair + 0.5 * cond + 0.5 * obj) if beta == 0.5 else (
    1.5 / (1.0 + beta) * (pair + beta * (cond + obj)))
support = -torch.logsumexp(s[:, train_mask], dim=-1).mean()
```

关键调度实现：

```python
for group in optimizer.param_groups:
    group['lr'] = eta * group['lr_scale']
```

新增的中心化修正 RMS 对应：

```math
\Delta U^{c}_{bji}=\Delta U_{bji}-\frac{1}{|\mathcal{A}|}\sum_{k\in\mathcal{A}}\Delta U_{bjk},\qquad
R_{\Delta}=\sqrt{\frac{1}{N|\mathcal{O}||\mathcal{A}|}\sum_{b,j,i}(\Delta U^{c}_{bji})^2}.
```

中心化去除条件 softmax 不敏感的属性公共平移。训练日志先按图像数累计中心化均方，再对 epoch 均方开根号；裁剪比例按实际发生裁剪的更新次数除以总更新次数计算。新诊断不参与模型梯度。

## 5. 超参数选取策略与实际实验矩阵

{{MATRIX}}

1. **初始探索：** seed3351 跑 A1–A6。A1/A2 与原 V1 比较 CP 学习率；A3/A4 与原 V1 比较 β；A5 联合降低 β 与 CP 学习率；A6 提供 β=0.1 的无 CP 对照。
2. **组合效果：** 原 V1、A1、A4、A5 构成学习率倍率 1/0.3 与 β=0.5/0.1 的 2×2 比较。各组从同 seed 的初始化重新训练。
3. **三 seed 确认：** A1/A5/A6 补 seed3352、3353，共新增六次训练。A2/A3/A4 保留单 seed 探索结果，统计时分开呈现。
4. **选模和设置取舍：** checkpoint 取首次严格最大 VAL CW AUC；设置比较结合 VAL CW/OW、未见条件属性及配对 seed 变化。TEST 用于报告本轮迁移表现。

| 固定项目 | 取值 |
|---|---|
| 框架与缓存 | 冻结 CLIP ViT-L/14，224px，768 维，复用既有特征 |
| 适配器 LR / adapter rank | 1e-3→1e-5 / 8 |
| rank / epochs / batch | 32 / 10 / 64 |
| ρ / κ / weight decay | 0.1 / 20 / 1e-4 |
| AdamW β₁、β₂ / ε | 0.9、0.999 / 1e-8 |
| 总梯度裁剪阈值 / τv | 1.0 / 0.07 |
| 本轮视觉门控与伙伴分支 | V1 静态 CP 及 V0 对照均关闭 |

## 6. 仓库结构与服务器命令

```text
VG-CPOD方案/
  v1/                  # 01 原实验
  02/
    README.md           # 本轮公式、代码修订、策略及末节数据分析
    code/README.md      # 训练源码与原版差异
    configs/README.md   # A1–A6、资产及参考指标
    results/README.md   # 12 次训练、48 份评价、24 份权重、预测、原导出包
    analysis/README.md  # 配对差值、类别变化、图表与复算
```

在仓库根目录恢复已运行的 A 组包：

```bash
git lfs pull
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 02 \
  --dest /data/xhw/vgcpod_project/replay_02
cd /data/xhw/vgcpod_project/replay_02/vgcpod_v1_roundA
conda activate vgcpod_v1
export VG_CACHE=/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14
export VG_A_OUT=/data/xhw/vgcpod_v1_runs/replay_02
export VG_A_SESSION=vgcpod_02_initial
bash scripts/start_round_a_tmux.sh --gpus 1 2 3 --seeds 3351
```

初始六组完成后，补齐本轮确认实验：

```bash
export VG_A_SESSION=vgcpod_02_confirm
bash scripts/start_round_a_tmux.sh \
  --gpus 1 2 3 --cases A1 A5 A6 --seeds 3352 3353
```

以上为复现命令。已有正式结果位于 [results](results/README.md)，复算统计使用：

```bash
python VG-CPOD方案/02/analysis/recompute.py
```

## 7. 数据分析与本轮结论

{{MAIN}}

均值 ± 三 seed 样本标准差；AUC 采用原保存的 `AUC_native`，单位为面积×100。

**保留 A5 作为下一轮主配置：β=0.1、CP LR=3e-4、适配器 LR=1e-3、rank32、10 epochs。** A1 保留为 β=0.5 参照，A6 作为同 β 的无 CP 对照。

- A1 相对原 V1 的 TEST CW AUC 平均提高 **2.203 点**，三个 seed 均提高，支持将 CP 学习率从 1e-3 降至 3e-4。
- A5 相对原 V1 的 CW/OW AUC 提高 **2.007 / 1.835 点**，相对 D0 提高 **4.151 / 0.699 点**；上述比较在三个 seed 均提高。
- A5 与 A1 的 TEST AUC 均值接近：CW −0.197、OW +0.021。A5 的 VAL OW 三 seed 均更高，TEST OW 标准差更小，未见条件属性准确率平均高 **6.011 点**。
- A5 相对 A6 的 TEST CW/OW 增量为 **4.325 / 0.471 点**；未见条件属性准确率为 **23.638% / 24.431%**。CP 平均在已见图像上多识别对约 56 张，在未见图像上少约 15 张，说明收益仍主要集中于已见组合。
- A5 的 TEST OW 固定 VAL bias HM 为 **40.638**，D0 为 **42.063**。精确 VAL bias 只将 A5 提高到 **40.764**，下一轮继续观察排序与固定工作点表现。

![三 seed 主结果](analysis/confirmed/figures/01_main.png)

本轮已完成支持损失与 CP 更新尺度的调整。下一轮直接在 A5 上比较混合读出、局部门控及弱局部门控，重点改善未见材质组合的迁移。详细配对结果与逐组合变化见 [数据分析 README](analysis/README.md)。
'''
    (STAGE / 'README.md').write_text(main.replace('{{MATRIX}}', matrix).replace('{{MAIN}}', main_table), encoding='utf-8')
    code = '''# 02 · 代码与修订\n\n''' + '''训练代码位于 [vgcpod_v1_roundA](vgcpod_v1_roundA/)，与服务器运行版本保持相同字节。配置、资产和发布支持文件在平行分区，通过仓库恢复命令还原完整运行包。\n\n''' + table(
        ['入口', '内容'], [[f'[{name}](vgcpod_v1_roundA/{file})', desc] for name, file, desc in [
            ('模型与损失', 'vgcpod/model.py', '归一化 β 目标、task/support 日志，前向沿用静态全局 CP'),
            ('训练与评测', 'vgcpod/engine.py', 'AdamW 参数分组、比例余弦、每轮 VAL CW/OW 与分布诊断'),
            ('CLI', 'run.py', '配置 variant 默认值、β 与 LR 倍率参数读取'),
            ('三卡队列', 'round_a.py', '六组初始探索、补 seed、逐运行汇总'),
            ('tmux 入口', 'scripts/start_round_a_tmux.sh', '指定三张物理 GPU 并保存 launcher 状态'),
            ('数学与集成检查', 'tests/', '原数学检查和本轮检查共 17 项')]]) + '''\n\n- [相对原 v1 的代码补丁](source_changes.patch)。\n- [服务器使用的完整冻结 ZIP](../results/archives/vgcpod_v1_roundA_20261019.zip)。\n- [原版检查记录](../results/package_validation/validation/summary.json)与[测试日志](../results/package_validation/validation/test_output.txt)。\n- [原始分析源码](original_analysis/)保持原字节；[仓库复算入口](../analysis/recompute.py)仅适配路径。\n- [成员映射](../package_members.json)记录恢复完整包时的原路径。\n\nCLIP 源码及其许可随代码归档：[CLIP_LICENSE](vgcpod_v1_roundA/vendor/CLIP_LICENSE)。模型公式与本轮修改说明见[主 README](../README.md)。\n'''
    (STAGE / 'code/README.md').write_text(code, encoding='utf-8')
    config_text = '# 02 · 配置与超参数策略\n\n' + matrix.replace('(configs/', '(') + '''\n\n- [默认配置](default.json)：保留原 β=0.5 与 CP 倍率 1，六组配置分别指定实验值。\n- [原 V0/V1 三 seed 参考](references/original_v1.json)：匹配特征缓存身份后计算配对差值。\n- [数据名称与索引](assets/)：标签、文本模板名称、历史输入摘要。\n- [12 次已完成运行的完整配置与指标](experiment_inventory.json)。\n\n先在 seed3351 比较六种设置，再对 A1/A5/A6 补两个 seed；所有运行均从相同 seed 初始化重新训练，同一任务中断时使用续训。固定项、推导及选择依据见[主 README 第 5 节](../README.md#5-超参数选取策略与实际实验矩阵)。下一轮主配置使用 A5，A1/A6 保留参照。\n'''
    (STAGE / 'configs/README.md').write_text(config_text, encoding='utf-8')
    result_text = '''# 02 · A 组训练结果\n\n2026-10-19：12 次训练均完成 10 epochs，共 48 份评价、48 份预测和 24 份 best/last 权重。A1/A5/A6 各三个 seed，A2/A3/A4 各一个 seed。\n\n''' + seed_table + '''\n\n每次运行按首次严格最大 VAL CW AUC_native 选模。HM 使用对应 VAL 的 bias，TEST 使用同一份选中权重。\n\n- [原始运行目录](raw_export/roundA_20261019/)：metrics、history、protocol、checkpoints、predictions、curves 与日志。\n- [初始六组 launcher 状态](initial_launch/)与[确认补跑状态](raw_export/roundA_20261019/round_a_status.json)。\n- [完整服务器导出包](archives/vgcpod_roundA_confirm_20261019.tar.gz)与[SHA256](archives/vgcpod_roundA_confirm_20261019.tar.gz.sha256)，由 Git LFS 保存。\n- [训练代码 ZIP](archives/vgcpod_v1_roundA_20261019.zip)与[SHA256](archives/vgcpod_v1_roundA_20261019.zip.sha256)。\n- [原版发布检查记录](package_validation/)与[原始说明文档字节](package_support/)。\n- [三 seed 数据分析](../analysis/README.md)和[文件 SHA256 清单](../file_manifest.json)。\n\n归档的服务器包已包含最初六组结果；seed3351 的 78 个模型成员与初次下载文件一致。A1/A5/A6 的多 seed 结论与 A2/A3/A4 的单 seed 探索结果分开报告。\n'''
    (STAGE / 'results/README.md').write_text(result_text, encoding='utf-8')
    pairs = []
    for a, o in [('Suede', 'Slippers'), ('Synthetic', 'Boots.Mid-Calf'), ('Suede', 'Boots.Mid-Calf'),
                 ('Canvas', 'Shoes.Loafers'), ('Canvas', 'Shoes.Flats'), ('Satin', 'Sandals')]:
        p = next(x for x in data['pairs'] if x['attribute'] == a and x['object'] == o)
        pairs.append([a + '–' + o, p['images'], f"{p['A5_cond_acc']:.2f}", f"{p['A6_cond_acc']:.2f}",
                      f"{p['A5_minus_A6']:+.2f}", f"{p['A5_minus_A6']*p['images']/100:+.1f}"])
    pairtable = table(['未见组合', '图像数', 'A5 条件属性准确率', 'A6 条件属性准确率', '差值', '正确图像数变化'], pairs)
    analysis = '''# 02 · 数据分析\n\n**A5 作为下一轮基座；A1/A6 作为已有参照。** 本页依据三个 seed 的配对结果分析学习率、支持损失和剩余迁移误差。AUC 为面积×100，准确率、HM 与差值均采用百分点；误差条为样本标准差。\n\n## 1. 三 seed 主结果\n\n''' + main_table + '''\n\n![主结果](confirmed/figures/01_main.png)\n\n## 2. 同 seed 配对差值\n\n''' + paired + '''\n\n![配对增量](confirmed/figures/02_paired.png)\n\nA1 证明降低 CP 学习率对 CW 有持续收益；A5 进一步恢复未见属性识别，其相对原 V1 的 OW AUC 增量在三个 seed 均为正。A5 与 A1 的 TEST AUC 均值接近，不能把单 seed 的 A5 优势推广为所有指标均胜出。A5 的 VAL OW 在三个 seed 均高于 A1，TEST OW 波动更小，因此继续以 A5 作为结构实验基座。\n\n## 3. 初始六组如何确定参数\n\n''' + seed_table + '''\n\nseed3351 中，A2 将 CP LR 进一步降至 1e-4 后，VAL CW 从 A1 的 42.246 降到 39.675；A3/A4 仅降低 β，未获得 A5 的组合表现。A5 相对 A4 只降低 CP LR，VAL CW/OW 提高 8.837 / 5.825 点，说明 β 与更新尺度需要联合考虑。原 V1、A1、A4、A5 的 2×2 比较及初始逐组差值保存在 [single_seed/tables](single_seed/tables/)。\n\n初始比较确定了 A1/A5/A6 三个确认配置；补 seed 后保留 A5 的 β=0.1、CP LR=3e-4。rank32、10 轮、batch64、适配器 LR=1e-3 沿用。\n\n## 4. CP 的收益和未见属性损失\n\n''' + diag_table + '''\n\n![已见支持质量和条件属性迁移](confirmed/figures/03_transfer.png)\n\nA5 相对 A1，将 TEST 未见图像的已见概率质量从 82.26% 降至 73.49%，未见条件属性准确率从 17.63% 升至 23.64%。相比同 β 的 A6，A5 未见条件属性准确率的三个 seed 差值为 −0.053 / −0.582 / −1.745 点；平均少识别对 15 张未见图像，已见图像则平均多识别对 56 张。\n\nVAL 未见条件属性准确率 A5/A6 为 21.81% / 16.05%，TEST 为 23.64% / 24.43%。这说明 CP 对不同未见组合的迁移程度不同。\n\n''' + pairtable + '''\n\n三类负向组合平均合计少识别对约 102.7 张；Canvas–Shoes.Loafers 多识别对 52 张，其他组合也有收益。这些变化相互抵消后，表现为未见总体少 15 张。下一轮优先检验局部视觉证据能否改善材质在不同物体间的迁移。\n\n## 5. OW 工作点与训练轮数\n\nA5 相对 D0 的 TEST OW AUC 提高 0.699 点，但固定 VAL bias 的 OW HM 为 40.638，对比 D0 的 42.063 仍低 1.425 点。精确 VAL bias 仅将 A5 提高至 40.764；其 TEST 曲线最高 HM 约为 41.780，D0 约为 43.215。下一轮同时记录 AUC 与固定 VAL bias 的 HM。\n\nA1 三个 seed 最佳轮为 8/6/6，A5 为 8/8/6；峰值到第 10 轮的 VAL CW 平均回落分别为 0.551 / 0.285 点，继续保持 10 轮。详细数值见 [history.csv](confirmed/tables/history.csv)。\n\n## 6. 下一轮投入方向\n\n以 A5 为基座，三个 GPU 分别运行：B1 全局与物体查询特征混合读出，B2 全局读出加局部门控，B3 全局读出加 η=0.25 的弱局部门控。先各跑 seed3351，再对 VAL 表现合适的配置补两个 seed。B 组作为下一轮实验，当前 02 归档记录 A 组已运行版本。\n\n## 7. 数据与复算\n\n- [三 seed 均值/标准差](confirmed/tables/mean_sd.csv)、[逐 seed 配对差值](confirmed/tables/paired_deltas.csv)。\n- [逐组合平均值](confirmed/tables/pair_mean.csv)、[逐 seed 逐组合](confirmed/tables/per_pair.csv)。\n- [已见/未见诊断](confirmed/tables/diagnostics.csv)、[全部评测指标](confirmed/tables/all_metrics.csv)。\n- [原分析与复核记录](confirmed/analysis.json)、[审计结果](confirmed/audit.json)。\n- [初始单 seed 结果](single_seed/)、[原始分析源码](../code/original_analysis/)。\n\n在仓库根目录执行 `python VG-CPOD方案/02/analysis/recompute.py`。需要 NumPy、PyTorch、Matplotlib；从已存预测复算 60 份评价，包括本轮 A1/A5/A6 与原 D0/V1，每模型四种设置，输出到 `analysis/recomputed/`。原脚本的数值计算保持不变，入口仅适配仓库路径。\n\n训练缓存图像数组继续由服务器现有缓存提供；统计与预测复算使用本仓库保存的权重、预测及缓存元数据。\n'''
    (STAGE / 'analysis/README.md').write_text(analysis, encoding='utf-8')


def finalize():
    owned = {p.relative_to(ROOT).as_posix() for p in STAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'recomputed' not in p.parts}
    manifest = [{'path': p.relative_to(STAGE).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}
                for p in sorted(STAGE.rglob('*')) if p.is_file() and p.name != 'file_manifest.json' and
                '__pycache__' not in p.parts and 'recomputed' not in p.parts]
    write(STAGE / 'file_manifest.json', manifest)
    owned.add('VG-CPOD方案/02/file_manifest.json')
    stages = read(ROOT / 'provenance/stages.json')
    stages = [s for s in stages if s.get('id') != '21']
    stages.append({'id': '21', 'line': 'VG-CPOD方案', 'number': '02', 'version': 'v1',
                   'title': '02 · ' + TITLE, 'path': 'VG-CPOD方案/02', 'primary_package': PACKAGE,
                   'status': '12次A组训练完成；48份评价、24份权重、预测及三seed数据分析已归档',
                   'completed_runs': 12, 'evaluations': 48, 'checkpoints': 24,
                   'confirmed_cases': ['A1', 'A5', 'A6'], 'confirmed_seeds': [3351, 3352, 3353],
                   'single_seed_cases': ['A2', 'A3', 'A4']})
    write(ROOT / 'provenance/stages.json', stages)
    owned.update(['README.md', 'VG-CPOD方案/README.md', 'tools/reproduce.py',
                  'tools/register_vgcpod_round_a.py', 'provenance/stages.json'])
    current = {r['path']: r for r in read(ROOT / 'provenance/current_manifest.json')}
    for rel in sorted(owned):
        p = ROOT / rel
        current[rel] = {'path': rel, 'sha256': sha(p), 'bytes': p.stat().st_size}
    write(ROOT / 'provenance/current_manifest.json', [current[p] for p in sorted(current)])


def check():
    entries = read(STAGE / 'file_manifest.json')
    for r in entries:
        p = STAGE / r['path']
        assert p.is_file() and sha(p) == r['sha256'], r['path']
    mapping = read(STAGE / 'package_members.json')['members']
    release = read(STAGE / mapping['release_manifest.json'])['files_sha256']
    for name, digest in release.items():
        assert sha(STAGE / mapping[name]) == digest, name
    with zipfile.ZipFile(STAGE / 'results/archives/vgcpod_v1_roundA_20261019.zip') as z:
        files = {n for n in z.namelist() if not n.endswith('/')}
        assert files == {PACKAGE + '/' + p for p in mapping}
        for name, target in mapping.items():
            assert z.read(PACKAGE + '/' + name) == (STAGE / target).read_bytes(), name
    raw = STAGE / 'results/raw_export/roundA_20261019'
    runs = list(raw.glob('A*/ut-zap50k/V*/seed*/metrics.json'))
    assert len(runs) == 12 and len(list(raw.rglob('predictions/*.npz'))) == 48
    assert len(list(raw.rglob('checkpoints/*.pt'))) == 24
    for p in runs:
        m, pr, h = read(p), read(p.parent / 'protocol.json'), read(p.parent / 'history.json')
        assert [r['epoch'] for r in h] == list(range(1, 11)) and h[-1]['step'] == 3600
        assert m['selected_epoch'] == max(range(10), key=lambda i: h[i]['val_cw']['AUC_native']) + 1
        assert sha(p.parent / 'checkpoints/best.pt') == m['checkpoint_sha256']
        assert pr['config'] == read(STAGE / 'configs/round_a' / (pr['config']['experiment_id'] + '.json'))
        for name, digest in pr['source_sha256'].items():
            assert sha(STAGE / mapping[name]) == digest
    broken = []
    for p in STAGE.rglob('README.md'):
        text = p.read_text(encoding='utf-8')
        assert text.count('```') % 2 == 0
        for link in re.findall(r'\]\(([^)\n]+)\)', text):
            link = link.split('#')[0]
            if link and not re.match(r'^[a-z]+://', link) and not (p.parent / link).exists():
                broken.append((p.relative_to(STAGE).as_posix(), link))
    assert not broken, broken
    for p in list((STAGE / 'code').rglob('*.py')) + [STAGE / 'analysis/recompute.py']:
        ast.parse(p.read_text(encoding='utf-8-sig'))
    return {'passed': True, 'files_verified': len(entries), 'package_members': len(mapping),
            'trained_runs': 12, 'evaluations': 48, 'checkpoints': 24, 'broken_links': broken}


def materialize(destination):
    destination = Path(destination).expanduser()
    if not destination.is_absolute():
        raise ValueError('Use an absolute destination')
    destination = destination.resolve()
    if destination.exists() or destination == ROOT or ROOT in destination.parents:
        raise FileExistsError(destination)
    mapping = read(STAGE / 'package_members.json')['members']
    for name, rel in mapping.items():
        copy(STAGE / rel, destination / PACKAGE / name)
    release = read(destination / PACKAGE / 'release_manifest.json')['files_sha256']
    assert all(sha(destination / PACKAGE / name) == digest for name, digest in release.items())
    write(destination / 'materialization.json', {'line': 'VG-CPOD方案', 'stage': '02',
          'package': PACKAGE, 'files': len(mapping), 'training_started': False})
    return {'line': 'VG-CPOD方案', 'stage': '02', 'package': str(destination / PACKAGE),
            'files_verified': len(mapping), 'training_started': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--import-data', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--materialize', type=Path)
    args = parser.parse_args()
    if args.import_data and args.source:
        import_data(args.source)
        result = check()
    elif args.finalize:
        finalize()
        result = check()
    elif args.check:
        result = check()
    elif args.materialize:
        result = materialize(args.materialize)
    else:
        parser.error('Choose --source PATH --import-data, --check, --finalize, or --materialize PATH')
    print(json.dumps(result, ensure_ascii=False))
