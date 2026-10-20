"""Archive experiment 03 and restore the exact B-group training package."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'VG-CPOD方案/03'
PACKAGE = 'vgcpod_v1_roundB'
TITLE = '读出解耦与局部弱门控'
ZIP = 'vgcpod_v1_roundB_20261019.zip'
TAR = 'vgcpod_roundB_results_20261019.tar.gz'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def relative_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or ':' in name:
        raise ValueError('Unexpected archive member: ' + name)
    return path


def import_data(source):
    source = Path(source).resolve()
    if STAGE.exists():
        raise FileExistsError(STAGE)
    STAGE.mkdir(parents=True)
    mapping = {}
    with zipfile.ZipFile(source / ZIP) as archive:
        for member in sorted(archive.namelist()):
            if member.endswith('/'):
                continue
            parts = relative_member(member).parts
            if parts[0] != PACKAGE:
                raise ValueError(member)
            name = PurePosixPath(*parts[1:]).as_posix()
            if name.startswith('configs/'):
                target = name
            elif name.startswith(('assets/', 'references/')):
                target = 'configs/' + name
            elif name.startswith('validation/') or name == 'release_manifest.json':
                target = 'results/package_validation/' + name
            elif Path(name).suffix in ('.md', '.html'):
                target = 'results/package_support/' + name + '.txt'
            else:
                target = 'code/' + PACKAGE + '/' + name
            destination = STAGE / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))
            mapping[name] = target
    write(STAGE / 'package_members.json', {'package': PACKAGE, 'members': mapping})
    for name in (ZIP, ZIP + '.sha256', TAR):
        copy(source / name, STAGE / 'results/archives' / name)
    archive_path = STAGE / 'results/archives' / TAR
    archive_path.with_name(TAR + '.sha256').write_text(sha(archive_path) + '  ' + TAR + '\n', encoding='ascii')
    raw_manifest, source_members = [], []
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = relative_member(member.name).parts
            if parts[0] == 'roundB_20261019' and parts[-1] != '.round_b.lock':
                payload = archive.extractfile(member).read()
                target = STAGE / 'results/raw_export' / PurePosixPath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                raw_manifest.append({'path': target.relative_to(STAGE).as_posix(),
                                     'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)})
            elif parts[0] == PACKAGE:
                name = PurePosixPath(*parts[1:]).as_posix()
                if name in mapping:
                    payload = archive.extractfile(member).read()
                    assert payload == (STAGE / mapping[name]).read_bytes(), member.name
                    source_members.append(name)
    write(STAGE / 'results/raw_export_members.json', raw_manifest)
    write(STAGE / 'results/server_source_identity.json', {'package': PACKAGE,
          'members_matched_to_deployment_zip': sorted(source_members),
          'matched_count': len(source_members), 'source_bytes_preserved': True})
    for folder in ('analysis_roundB_results_20261019',):
        for path in sorted((source / folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.csv','.json','.png','.txt'):
                copy(path, STAGE / 'analysis/pilot' / path.relative_to(source / folder))
    for folder in ('analysis_roundB_results_20261019', 'analysis_roundB_20261019'):
        for path in sorted((source / folder).iterdir()):
            if path.is_file() and path.suffix in ('.py', '.cjs'):
                copy(path, STAGE / 'code/original_analysis' / folder / path.name)
    for path in sorted(source.glob('VG-CPOD_v1_B组*.md')):
        copy(path, STAGE / 'results/package_support/original_reports' / (path.name + '.txt'))
    create_patch()
    create_recompute(source)
    generate_readmes()
    inventory = []
    for path in sorted((STAGE / 'results/raw_export/roundB_20261019').glob('B*/ut-zap50k/V*/seed*/metrics.json')):
        saved, protocol = read(path), read(path.parent / 'protocol.json')
        inventory.append({'case': protocol['config']['experiment_id'], 'seed': protocol['seed'],
            'selected_epoch': saved['selected_epoch'], 'completed_epochs': 10,
            'path': path.parent.relative_to(STAGE).as_posix(),
            'config': protocol['config'], 'metrics': saved['metrics']})
    write(STAGE / 'configs/experiment_inventory.json', inventory)
    write(STAGE / 'runbook.json', {'line': 'VG-CPOD方案', 'stage': '03', 'primary_package': PACKAGE,
        'commands': [{'source': 'README.md', 'language': lang, 'text': text}
                     for lang, text in re.findall(r'```(bash|powershell)\n(.*?)```', (STAGE / 'README.md').read_text(encoding='utf-8'), re.S)]})
    write(STAGE / 'external_inputs.json', {'framework': 'frozen CLIP ViT-L/14',
        'cache': '/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14',
        'feature_manifest_sha256': 'bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a',
        'feature_arrays': 'Reuse the existing frozen CLIP server cache; raw images and CLIP weights rebuild it.',
        'scalar_references': 'configs/references/round_a_confirm.json',
        'paired_prediction_references': '../02/results/raw_export/roundA_20261019',
        'cache_metadata': '../v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14',
        'original_input_index': '../v1/external_inputs.json'})
    write(STAGE / 'inputs.json', {'line': 'VG-CPOD方案', 'stage': '03', 'version': 'v1',
        'title': TITLE, 'knowledge': [], 'completed_runs': 3, 'model_epochs': 30, 'evaluations': 12,
        'checkpoints': 6, 'predictions': 12, 'completed_cases': ['B1','B2','B3'], 'completed_seeds': [3351],
        'package_member_map': 'package_members.json', 'experiment_inventory': 'configs/experiment_inventory.json',
        'code_files': [p.relative_to(STAGE).as_posix() for p in sorted((STAGE / 'code').rglob('*')) if p.is_file()],
        'configs': [p.relative_to(STAGE).as_posix() for p in sorted((STAGE / 'configs').rglob('*.json'))],
        'documents': [p.relative_to(STAGE).as_posix() for p in sorted(STAGE.rglob('README.md'))]})
    (STAGE / '.gitignore').write_text('analysis/recomputed/\n', encoding='utf-8')
    finalize()


def create_patch():
    before = ROOT / 'VG-CPOD方案/02'
    old_map = read(before / 'package_members.json')['members']
    new_map = read(STAGE / 'package_members.json')['members']
    patches = []
    for name, target in sorted(new_map.items()):
        if not (name.endswith(('.py','.sh')) or name.startswith('configs/round_b/')):
            continue
        previous = (before / old_map[name]).read_text(encoding='utf-8') if name in old_map else ''
        current = (STAGE / target).read_text(encoding='utf-8')
        patches.extend(difflib.unified_diff(previous.splitlines(keepends=True), current.splitlines(keepends=True),
            fromfile='a/' + name if name in old_map else '/dev/null', tofile='b/' + name))
    (STAGE / 'code/source_changes.patch').write_text(''.join(patches), encoding='utf-8')


def create_recompute(source):
    original = (source / 'analysis_roundB_results_20261019/analyze_b.py').read_text(encoding='utf-8')
    start = original.index('ROOT = Path(')
    end = original.index('CASES =', start)
    original = original[:start] + """STAGE = Path(__file__).resolve().parent.parent
ROOT = STAGE.parent.parent
OUT = Path(__file__).resolve().parent / 'recomputed'
BR = STAGE / 'results/raw_export/roundB_20261019'
BS = STAGE / 'code/vgcpod_v1_roundB'
AR = STAGE.parent / '02/results/raw_export/roundA_20261019'
OLD = STAGE.parent / 'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs'
""" + original[end:]
    replacements = {
        "ROOT / 'analysis_20261018/analyze_results.py'": "STAGE.parent / 'v1/code/analysis_20261018/analyze_results.py'",
        "ROOT / 'vgcpod_roundB_results_20261019.tar.gz'": "STAGE / 'results/archives/vgcpod_roundB_results_20261019.tar.gz'",
        "ROOT / 'vgcpod_v1_roundB/configs/round_b'": "STAGE / 'configs/round_b'",
        "sha(BS / name) == fingerprint == sha(ROOT / 'vgcpod_v1_roundB' / name)": "sha(BS / name) == fingerprint",
    }
    for before, after in replacements.items():
        assert before in original, before
        original = original.replace(before, after)
    (STAGE / 'analysis/recompute.py').write_text(
        '# Repository paths adapted; archived numerical calculations are unchanged.\n' + original, encoding='utf-8')


def generate_readmes():
    data = read(STAGE / 'analysis/pilot/analysis.json')
    metric = lambda c,s: next(r for r in data['metrics'] if r['case']==c and r['setting']==s)
    diag = lambda c: next(r for r in data['diagnostics'] if r['case']==c and r['phase']=='test')
    gate = lambda c: next(r for r in data['gates'] if r['case']==c and r['chosen'])
    delta = lambda c,s: next(r for r in data['comparisons'] if r['comparison']==c+'-A5' and r['setting']==s)
    main_table = table(['组','VAL CW AUC','VAL OW AUC','TEST CW AUC','TEST OW AUC','TEST 未见条件属性'],
        [[c] + [f"{metric(c,s)['AUC_native']:.3f}" for s in ('val_cw','val_ow','test_cw','test_ow')] +
         [f"{diag(c)['attr_unseen']:.3f}%"] for c in ('A5','A6','B1','B2','B3')])
    deltas = table(['相对 A5','VAL CW','VAL OW','TEST CW','TEST OW','TEST 未见属性','TEST OW HM'],
        [[c] + [f"{delta(c,s)['AUC_native']:+.3f}" for s in ('val_cw','val_ow','test_cw','test_ow')] +
         [f"{delta(c,'test_ow')['gt_object_attr_acc_unseen']:+.3f}",f"{delta(c,'test_ow')['HM_at_val_bias']:+.3f}"]
         for c in ('B1','B2','B3')])
    matrix = table(['组','修正读出','门控输入','η / 系数范围','已完成 seed'],[
        ['[B1](configs/round_b/B1.json)','归一化的 0.5 全局 + 0.5 物体查询特征','关闭','全 1','3351'],
        ['[B2](configs/round_b/B2.json)','全局 g','物体查询特征','1 / (0,2)','3351'],
        ['[B3](configs/round_b/B3.json)','全局 g','物体查询特征','0.25 / (0.75,1.25)','3351']])
    transfer = table(['组','已见条件属性','未见条件属性','未见正确图像数','未见图像的已见概率质量','物体准确率'],
        [[c,f"{diag(c)['attr_seen']:.3f}%",f"{diag(c)['attr_unseen']:.3f}%",diag(c)['unseen_correct'],
          f"{diag(c)['support_unseen']:.3f}%",f"{diag(c)['object_acc']:.3f}%"] for c in ('A5','A6','B1','B2','B3')])
    gates = table(['组','选中 epoch','γ 均值','批内总标准差','跨图像变化尺度','sigmoid 均值','ρ×γ 均值'],
        [[c,gate(c)['epoch'],f"{gate(c)['gamma_mean']:.6f}",f"{gate(c)['gamma_std']:.6f}",
          f"{gate(c)['gamma_image_std']:.6f}",f"{gate(c)['raw_sigmoid_mean']:.4f}",
          f"{gate(c)['effective_scalar_rho']:.6f}"] for c in ('B2','B3')])
    main = r'''# 03 · 读出解耦与局部弱门控

**B 组实验，VG-CPOD v1，训练日期 2026-10-19，归档日期 2026-10-20。** 框架为冻结 CLIP ViT-L/14。UT-Zappos natural split，B1/B2/B3 各完成 seed3351、10 epochs，共 3 次训练、30 个模型 epoch、12 份 VAL/TEST CW/OW 评价及预测、6 份 best/last 权重。

[代码与具体改动](code/README.md) · [超参数配置](configs/README.md) · [全部训练结果](results/README.md) · [数据分析](analysis/README.md) · [02 A 组基座](../02/README.md)

## 1. 实验目的与修订范围

02 确定了 A5 的 β=0.1、CP LR=3e-4。静态 CP 的组合 AUC 有提升，未见条件属性识别仍低于同 β 的 A6。本轮研究局部视觉证据怎样进入 CP：B1 改变修正读出，B2/B3 保留全局读出，由物体查询特征调节 CP 模式，并比较完整与较弱的调制幅度。

CLIP 编码、已有特征缓存、物体 logits、基础条件 logits、概率链式分解，以及 A5 的训练损失和分组学习率继续沿用。新增修改集中在修正读出与门控输入的分离、门控幅度 η、门控变化统计和三 GPU 队列。

## 2. 公式修订：分开修正读出与门控输入

记全局特征为 $g_b\in\mathbb{R}^{d}$，物体查询特征为 $f^o_{bj}\in\mathbb{R}^{d}$，本轮 $d=768$。读出特征定义为：

```math
\begin{aligned}
\operatorname{unit}(z)&=\frac{z}{\max(\lVert z\rVert_2,10^{-12})},\\
h_{bj}^{\mathrm{B1}}&=\operatorname{unit}\!\left((1-\alpha)g_b+\alpha f^o_{bj}\right),\quad\alpha=0.5,\\
h_{bj}^{\mathrm{B2}}&=h_{bj}^{\mathrm{B3}}=g_b.
\end{aligned}
```

α 表示局部特征的权重。混合特征只进入 CP 修正读出；物体分支和基础条件分支继续读取全局 g。定义文本模式：

```math
u_i=A_u e_i^a\in\mathbb{R}^{r},\qquad
v_j=A_v e_j^o\in\mathbb{R}^{r},\qquad W_c\in\mathbb{R}^{d\times r},\quad r=32.
```

修正统一为：

```math
\begin{aligned}
\Delta U_{bji}&=\rho\kappa\,h_{bj}^{\top}W_c
\left(u_i\odot v_j\odot\gamma_{bj}\right),\\
U_{bji}&=U_{bji}^{\mathrm{base}}+\Delta U_{bji},\\
\log p(a_i,o_j\mid x_b)&=\log\operatorname{softmax}(S_b)_j
+\log\operatorname{softmax}(U_{bj:})_i.
\end{aligned}
```

$S_b$ 为物体 logits。联合概率在全部物体与属性组合上归一化，张量轴为 `[batch, object, attribute]`。给定门控和物体后，等价算子为 $W_c\operatorname{diag}(v_j\odot\gamma_{bj})A_u$，秩至多为 r。Wc 初始为零，三个实验都从相同 seed 的零修正开始。

## 3. 公式修订：以 η 控制局部模式调制幅度

B1 使用全 1 模式系数。B2/B3 从局部特征计算：

```math
\begin{aligned}
\gamma_{bj}^{(\eta)}
&=\mathbf{1}_{r}+\eta\left[
2\sigma\!\left(\frac{B_{\mathrm{vis}}f^o_{bj}}{\tau_{\gamma}}\right)-\mathbf{1}_{r}\right],
\qquad B_{\mathrm{vis}}\in\mathbb{R}^{r\times d},\\
\eta_{\mathrm{B2}}&=1,\qquad \eta_{\mathrm{B3}}=0.25,\qquad \tau_{\gamma}=0.1.
\end{aligned}
```

η=1 恢复原 $2\sigma(\cdot)$，各模式系数位于 (0,2)；η=0.25 时位于 (0.75,1.25)；η=0 恢复全 1。在相同参数和输入下，CP 收缩对门控线性，因此：

```math
\Delta U_{\eta=0.25}=0.75\,\Delta U_{\eta=0}+0.25\,\Delta U_{\eta=1}.
```

本轮沿用 02 的归一化目标：

```math
\begin{aligned}
\ell_{\mathrm{task}}(\beta)
&=\frac{1.5}{1+\beta}\left[\ell_{\mathrm{pair}}
+\beta(\ell_{\mathrm{cond}}+\ell_{\mathrm{obj}})\right]\\
&=1.5\ell_{\mathrm{pair}}+\frac{1.5\beta}{1+\beta}\ell_{\mathrm{support}},\qquad\beta=0.1.
\end{aligned}
```

支持损失系数保持为 0.13636。B2/B3 的总目标加入现有正交项：

```math
\ell_{\mathrm{B2/B3}}=\ell_{\mathrm{task}}
+\frac{0.001}{r^2}\left\lVert B_{\mathrm{vis}}B_{\mathrm{vis}}^{\top}-I_r\right\rVert_F^2.
```

B1 的目标为 task。三个实验均关闭伙伴与原子属性辅助分支。

## 4. 具体代码修订

| 文件 | 已实施的修改 | 实验作用 |
|---|---|---|
| [model.py](code/vgcpod_v1_roundB/vgcpod/model.py) | `readout_feature` 与 `gate_input` 独立配置；加入 `readout_alpha` | 分开改变读出与模式调制，保留原变体默认计算 |
| 同上 | `gate_input` 决定 Bvis 是否训练；`gate_eta` 控制模式系数幅度 | B1 静态，B2/B3 动态；η=1 保留原浮点运算顺序 |
| [engine.py](code/vgcpod_v1_roundB/vgcpod/engine.py) | 增加 `gamma_image_variance`、`gamma_abs_deviation` | 区分图像变化和整体偏离 1 的程度 |
| [round_b.py](code/vgcpod_v1_roundB/round_b.py) | 独立三 GPU 队列、续训、跳过已完成运行；A5/A6 同 seed 配对及配置聚合 | 三组结构实验及结果汇总 |
| [scripts](code/vgcpod_v1_roundB/scripts/) | tmux 内使用当前环境的绝对 Python 路径，保存退出码 | GPU1/2/3 分别运行 B1/B2/B3 |
| [test_round_b.py](code/vgcpod_v1_roundB/tests/test_round_b.py) | 特征干预、η 插值、算子秩、梯度、训练导出与中断续训 | 与原测试合计 22 项检查 |

三组内部 `variant` 均为 V1，实际读出与动态机制由配置记录。B2/B3 的 Bvis 与 Au/Av/Wc 同属 CP 参数组，使用 3e-4→3e-6 的学习率。

核心实现对应：

```python
raw_gamma = 2 * torch.sigmoid((gate_input @ self.Bvis.T) / c['tau_gamma'])
gamma = raw_gamma if self.gate_eta == 1 else 1 + self.gate_eta * (raw_gamma - 1)
modes = (readout @ self.Wc) * gamma * v[None, :, :]
cond = base + rho * kappa * torch.einsum('bor,ir->boi', modes, u)
```

训练日志的新增跨图像方差为：

```math
\begin{aligned}
\mu_{tjk}&=\frac{1}{n_t}\sum_{b\in\mathcal{B}_t}\gamma_{bjk},\\
V_{\mathrm{img}}&=\frac{1}{\sum_t n_t}\sum_t n_t\,
\frac{1}{|\mathcal{O}|r}\sum_{j,k}\frac{1}{n_t}
\sum_{b\in\mathcal{B}_t}(\gamma_{bjk}-\mu_{tjk})^2,\\
\sigma_{\mathrm{img}}&=\sqrt{V_{\mathrm{img}}}.
\end{aligned}
```

这里 t 遍历选定 epoch 内的训练 batch。方差使用 `unbiased=False`，按图像数累计；分析报告中的跨图像变化尺度是该 epoch 日志的平方根。`gamma_abs_deviation` 同样按图像数累计各系数与 1 的绝对差均值。诊断计算从梯度图分离。

原变体默认配置的四次优化更新与 A 组实现逐位一致，记录见[兼容性检查](results/package_validation/validation/legacy_compatibility.json)。完整差异见[代码补丁](code/source_changes.patch)。

## 5. 超参数选取策略与实际实验矩阵

{{MATRIX}}

1. **采用 02 已确认的训练基座。** 固定 β=0.1、CP LR=3e-4、适配器 LR=1e-3，结构比较集中于读出与门控。
2. **B1 比较读出。** 取等权混合 α=0.5，与全局静态 A5 比较局部信息进入修正的效果。
3. **B2 比较输入通路。** 保留 A5 全局读出，让物体查询特征只产生模式系数。
4. **B3 比较幅度。** 相对 B2 只将 η 从 1 降为 0.25，保持相同的局部输入、损失和学习率。
5. **运行与选模。** 三组均从 seed3351 初始化训练 10 轮，checkpoint 按首次严格最大 VAL CW AUC 选取；同 seed A5/A6 提供静态 CP 和无 CP 参照。配置比较同时观察 VAL OW 与未见条件属性，TEST 报告迁移表现。

| 固定项目 | 取值 |
|---|---|
| CLIP / 特征 | 冻结 ViT-L/14，224px，768 维，复用现有缓存 |
| 适配器 LR / CP LR | 1e-3→1e-5 / 3e-4→3e-6，共 3600 次更新 |
| β / rank / adapter rank | 0.1 / 32 / 8 |
| epochs / batch | 10 / 64 |
| ρ / κ / τv / τγ | 0.1 / 20 / 0.07 / 0.1 |
| AdamW β₁、β₂ / ε / weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪阈值 / 动态正交项权重 | 1.0 / 0.001 |

## 6. 仓库结构与复现命令

```text
03/
  README.md          # 公式、代码修订、超参策略和末节数据分析
  code/              # 冻结训练源码、原分析源码、相对02补丁
  configs/           # B1/B2/B3、兼容配置、资产、A组参考
  results/           # 3次训练、预测、权重、日志和原始归档
  analysis/          # 同seed分析、图表和仓库复算入口
```

在仓库根目录恢复本轮实际运行包：

```bash
git lfs pull
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 03 \
  --dest /data/xhw/vgcpod_project/replay_03
cd /data/xhw/vgcpod_project/replay_03/vgcpod_v1_roundB
conda activate vgcpod_v1
export VG_CACHE=/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14
export VG_A_OUT=/data/xhw/vgcpod_v1_runs/roundA_20261019
export VG_B_OUT=/data/xhw/vgcpod_v1_runs/replay_03
export VG_B_SESSION=vgcpod_03
bash scripts/start_round_b_tmux.sh --gpus 1 2 3 --cases B1 B2 B3 --seeds 3351
```

配置和原包支持文件恢复后保持原字节，训练直接使用已有 CLIP 缓存。已有本轮结果位于 [results](results/README.md)。从保存预测复算数据分析：

```bash
python VG-CPOD方案/03/analysis/recompute.py
```

## 7. 数据分析与本轮结论

{{MAIN}}

本表全部为 seed3351，A5/A6 来自 02；AUC 为面积×100，条件属性准确率给定真实物体后在全部属性中预测。B1/B2/B3 最佳 epoch 分别为 10/5/8。

**A5 继续作为主基座，B3 保留为小幅组合增益参照。** B3 相对 A5 的 VAL CW/OW AUC 提高 0.334 / 0.416 点，TEST CW/OW 提高 **0.308 / 0.148 点**；TEST 未见属性少识别对 5 张，OW 固定 VAL bias 的 HM 变化为 −0.067 点。

- B1 的 VAL/TEST CW/OW AUC 均低于 A5，本轮等权混合读出没有改善组合与属性表现。
- B2 的 VAL CW/OW AUC 提高 0.861 / 0.831 点，TEST OW 降低 1.512 点，未见属性少识别对 29 张。
- B3 在未见图像上修正 7 张 A5 的错误，同时将 12 张原本正确的图像变为错误；Suede–Slippers、Suede–Boots.Mid-Calf、Synthetic–Boots.Mid-Calf 合计少识别对 10 张，其他五个组合各增加 1 张。
- B3 选中 epoch 的训练门控均值为 **1.240767**，接近上界 1.25；跨图像变化尺度为 **0.002861**。B2 对应均值 1.978014、变化尺度 0.004859。两组都明显向共同放大移动。

![本轮同seed结果](analysis/pilot/figures/01_results.png)

这些门控统计支持进一步区分静态修正幅度与局部模式选择。后续方向为静态 1.25 倍修正对照、均值为 1 的弱门控，以及 B3 重复 seed。上述属于后续实验设计；03 保存本轮已经训练的原门控公式。详细门控证据、逐组合变化与下一步公式见[数据分析 README](analysis/README.md)。
'''
    (STAGE / 'README.md').write_text(main.replace('{{MATRIX}}',matrix).replace('{{MAIN}}',main_table),encoding='utf-8')
    code = '# 03 · 代码与修订\n\n' + '''训练代码位于 [vgcpod_v1_roundB](vgcpod_v1_roundB/)，保持服务器运行版本的字节。配置、资产和发布支持文件存放在平行分区，通过仓库恢复命令还原完整运行包。\n\n''' + table(
        ['入口','内容'], [[f'[{name}](vgcpod_v1_roundB/{path})',desc] for name,path,desc in [
        ('模型与损失','vgcpod/model.py','独立读出与门控配置；混合归一化；η幅度；沿用A5目标'),
        ('训练与评测','vgcpod/engine.py','沿用分组学习率与评价，新增跨图像门控方差和偏离1的均值'),
        ('B组三卡队列','round_b.py','B1/B2/B3独立训练，续训、跳过完成运行，按配置及同seed配对汇总'),
        ('CLI','run.py','沿用A组训练、evaluate与diagnose命令'),
        ('tmux入口','scripts/start_round_b_tmux.sh','向tmux传入当前conda环境的绝对Python路径'),
        ('数学与集成检查','tests/test_round_b.py','新增5项B组检查，与原检查合计22项')]]) + '''

## 具体改动

- `readout_feature` 支持 global/local/mixed；α 控制 mixed 中局部特征的权重。
- `gate_input` 支持 none/global/local，与修正读出独立；本轮 B2/B3 从 local 生成门控。
- `gate_eta` 控制相对 1 的调制幅度；η=1 使用原计算顺序。
- `gamma_image_variance` 在每个物体/模式上沿 batch 图像轴计算，`gamma_abs_deviation` 记录共同偏移。
- `round_b.py` 输出逐运行指标、配置均值/样本标准差、同 seed A5/A6 与 B3−B2 增量及逐组合差值。

## 原版与复算入口

- [相对02训练包的代码及B配置补丁](source_changes.patch)。
- [完整冻结ZIP](../results/archives/vgcpod_v1_roundB_20261019.zip)、[成员映射](../package_members.json)。
- [22项测试日志](../results/package_validation/validation/test_output.txt)、[真实CLI三进程CPU运行记录](../results/package_validation/validation/cli_output.txt)。
- [原九变体四次更新逐位对照](../results/package_validation/validation/legacy_compatibility.json)、[本地验证环境与结果](../results/package_validation/validation/result.json)。
- [服务器源码与ZIP一致性](../results/server_source_identity.json)。
- [原始分析源码](original_analysis/)保持原字节；[仓库复算入口](../analysis/recompute.py)仅适配路径。

CLIP 源码与许可随包保存：[CLIP_LICENSE](vgcpod_v1_roundB/vendor/CLIP_LICENSE)。本轮已实施公式和日志定义见[主README](../README.md)。
'''
    (STAGE / 'code/README.md').write_text(code,encoding='utf-8')
    config = '# 03 · 配置与超参数策略\n\n' + matrix.replace('(configs/','(') + '''

## 选取策略

02 的三 seed 结果确定 A5 作为训练基座。本轮固定 β、学习率、rank、轮数和 batch，将比较集中于读出路径及 η：B1 的等权混合对应 α=0.5；B2 使用原 η=1；B3 将 η 降为 0.25，保持全局读出和相同局部输入。

三组均为 seed3351、10 epochs，CP LR=3e-4→3e-6，适配器 LR=1e-3→1e-5，β=0.1，rank32，adapter rank8，batch64，ρ=0.1，κ=20，τγ=0.1，weight decay=1e-4。B2/B3 的 Bvis 使用 CP 参数组及 λorth=0.001。

## 配置与资产索引

- [完整已完成运行配置与指标](experiment_inventory.json)。
- [默认配置](default.json)与[02兼容配置](round_a/)保留为原运行包成员；03 的实际运行入口是 round_b/B1、B2、B3。
- [A1/A5/A6三seed标量参考](references/round_a_confirm.json)与[原V0/V1参考](references/original_v1.json)。
- [标签、文本命名与数据索引](assets/)；图像特征复用已有冻结 CLIP 缓存。

三组内部均采用 V1 标识，由 `readout_feature`、`gate_input` 与 `gate_eta` 决定实际计算。原子属性及伙伴分支关闭，配置中的对应历史权重不参与本轮目标。完整固定项与选择依据见[主README第5节](../README.md#5-超参数选取策略与实际实验矩阵)。
'''
    (STAGE / 'configs/README.md').write_text(config,encoding='utf-8')
    result_rows = [[c,3351,metric(c,'test_cw')['epoch']] + [f"{metric(c,s)['AUC_native']:.4f}" for s in ('val_cw','val_ow','test_cw','test_ow')]
                   for c in ('B1','B2','B3')]
    results = '# 03 · B组训练结果\n\n2026-10-19：B1/B2/B3 各完成 seed3351、10 epochs，共3次训练、30个模型epoch、12份评价与预测、6份best/last权重。\n\n' + table(
        ['组','seed','最佳epoch','VAL CW','VAL OW','TEST CW','TEST OW'],result_rows) + '''

选中权重由首次严格最大 VAL CW AUC_native 决定，使用同一权重评价 VAL/TEST 的 CW/OW。TEST HM 使用对应 VAL 的 bias。

- [原始运行目录](raw_export/roundB_20261019/)：metrics、history、protocol、checkpoints、predictions、curves和日志。
- [队列完成状态](raw_export/roundB_20261019/round_b_status.json)、[退出码](raw_export/roundB_20261019/launcher.exitcode)。
- [逐运行汇总](raw_export/roundB_20261019/round_b_summary.csv)、[原导出同seed配对增量](raw_export/roundB_20261019/round_b_paired_deltas.csv)、[原逐组合增量](raw_export/roundB_20261019/per_pair_gains.csv)。
- [服务器完整导出包](archives/vgcpod_roundB_results_20261019.tar.gz)与[SHA256](archives/vgcpod_roundB_results_20261019.tar.gz.sha256)。
- [实际训练代码ZIP](archives/vgcpod_v1_roundB_20261019.zip)与[SHA256](archives/vgcpod_v1_roundB_20261019.zip.sha256)。二进制归档、权重及预测由 Git LFS 保存。
- [原版发布验证](package_validation/)、[原说明文档字节](package_support/)、[服务器源码身份](server_source_identity.json)。
- [文件清单](raw_export_members.json)、[阶段SHA256清单](../file_manifest.json)、[完整数据分析](../analysis/README.md)。

每组结果位于 `raw_export/roundB_20261019/B*/ut-zap50k/V1/seed3351/`；已有A5/A6参照位于[02的原始运行目录](../../02/results/raw_export/roundA_20261019/)。
'''
    (STAGE / 'results/README.md').write_text(results,encoding='utf-8')
    pair_rows = []
    for a,o in [('Suede','Slippers'),('Suede','Boots.Mid-Calf'),('Synthetic','Boots.Mid-Calf')]:
        row = next(r for r in data['pair_deltas'] if r['comparison']=='B3-A5' and r['phase']=='test' and r['attribute']==a and r['object']==o)
        ref = next(r for r in data['pair_deltas'] if r['comparison']=='B3-A6' and r['phase']=='test' and r['attribute']==a and r['object']==o)
        pair_rows.append([a+'–'+o,row['images'],f"{row['right_acc']:.2f}",f"{row['left_acc']:.2f}",
                          f"{ref['right_acc']:.2f}",f"{row['correct_delta']:+d}"])
    pairs = table(['未见组合','图像数','A5','B3','A6','B3−A5正确图像数'],pair_rows)
    analysis = '# 03 · 数据分析\n\n' + '''**A5继续作为主基座，B3保留为小幅组合增益参照。** 本页使用seed3351配对分析B1/B2/B3与已有A5/A6。AUC为面积×100，准确率、HM和差值采用百分点。\n\n## 1. 同seed主结果\n\n''' + main_table + '''

![同seed结果](pilot/figures/01_results.png)

## 2. 相对A5的配对增量

''' + deltas + '''

B1的等权混合读出降低了本轮VAL/TEST组合AUC与未见属性表现。B2提高VAL CW/OW AUC，但VAL未见条件属性从A5的20.924%降到19.897%，TEST OW下降1.512点。B3的VAL与TEST组合AUC都有小幅提高，未见条件属性仍略低于A5。

B3的TEST CW/OW固定VAL bias HM为48.474 / 43.120，A5为48.407 / 43.187；工作点表现接近原水平。未见组内排序准确率由50.291%升至50.767%，与AUC的小幅改善一致。

## 3. 未见属性与具体组合

''' + transfer + '\n\n' + pairs + '''

条件属性准确率给定真实物体，在全部属性中预测。B3与A5的已见正确图像数相同；在1,891张未见图像中恢复7张、退化12张，净减少5张。三个重点负向组合合计减少10张，另外五个组合各增加1张。B3与A5的TEST条件属性top-1预测一致率为98.730%。

![未见组合变化](pilot/figures/03_unseen_changes.png)

## 4. 门控接近共同放大的证据

''' + gates + r'''

表中γ统计来自选中epoch的训练batch，跨图像变化尺度为沿图像轴求方差、按图像数累计后开平方，定义见[主README](../README.md#4-具体代码修订)。

B2/B3的归一化平均偏移分别达到0.9780 / 0.9631，接近最大偏移1；B3平均偏移0.24077，跨图像变化尺度0.00286，相差约84倍。Bvis投影行的平均非对角余弦相似度分别为0.264 / 0.183，初始化时各行正交。这些观察支持“共同放大加较小图像变化”的解释。

选中epoch的梯度裁剪步骤比例：A5为17.78%，B2为50.28%，B3为29.44%。

![学习曲线与门控](pilot/figures/02_learning_and_gate.png)

## 5. 本轮之后的方向

后续优先比较静态幅度与模式相对调制：C1采用A5静态CP，将ρ从0.1改为0.125；C2保留全局读出与η=0.25，在模式轴去除共同偏移；另用一张卡补B3的seed3352。三个实验构成下一轮计划。

常数门控 $\gamma=c\mathbf{1}$ 在相同参数下等价于把ρ乘以c，因此C1直接测量1.25倍修正的收益。C2候选为：

```math
\begin{aligned}
q_{bjk}&=2\sigma\!\left(\frac{(B_{\mathrm{vis}}f^o_{bj})_k}{\tau_{\gamma}}\right)-1,\qquad
\bar q_{bj}=\frac{1}{r}\sum_k q_{bjk},\\
\gamma_{bjk}^{\mathrm{C2}}&=1+\frac{\eta}{2}(q_{bjk}-\bar q_{bj}),\qquad\eta=0.25,\\
\frac{1}{r}\sum_k\gamma_{bjk}^{\mathrm{C2}}&=1,\qquad
\gamma_{bjk}^{\mathrm{C2}}\in(0.75,1.25).
\end{aligned}
```

均值恒等式来自中心化后的和为零，范围来自q与其均值之差位于(-2,2)。C2将局部信息的作用集中在模式间的相对分配。03的训练代码与结果仍采用本轮原始门控；上述公式用于解释后续研究方向。

β=0.1、CP LR=3e-4、适配器LR=1e-3、rank32、10轮、batch64与τγ=0.1继续作为后续基座。

## 6. 数据与复算

- [全部指标](pilot/tables/all_metrics.csv)、[同seed增量](pilot/tables/comparisons.csv)。
- [条件属性修正/退化图像数](pilot/tables/conditional_changes.csv)、[逐组合增量](pilot/tables/per_pair_deltas.csv)。
- [已见/未见诊断](pilot/tables/diagnostics.csv)、[各轮历史](pilot/tables/history.csv)、[门控统计](pilot/tables/gates.csv)、[Bvis几何统计](pilot/tables/gate_geometry.csv)。
- [结构化分析](pilot/analysis.json)、[原复算记录](pilot/audit.json)、[原分析源码](../code/original_analysis/)。

在仓库根目录执行：

```bash
python VG-CPOD方案/03/analysis/recompute.py
```

需要NumPy、PyTorch和Matplotlib。入口从本轮B1/B2/B3、02的A1/A5/A6以及01的D0/V1读取seed3351预测，共复算32份评价，输出到`analysis/recomputed/`。原始分析的数值计算保持不变，仅适配仓库路径；native/exact AUC与固定VAL bias HM的原复算误差均为0。
'''
    (STAGE / 'analysis/README.md').write_text(analysis,encoding='utf-8')


def finalize():
    files = sorted(p for p in STAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'recomputed' not in p.parts)
    manifest = [{'path':p.relative_to(STAGE).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
                for p in files if p.name != 'file_manifest.json']
    write(STAGE / 'file_manifest.json',manifest)
    owned = {p.relative_to(ROOT).as_posix() for p in files}
    owned.add('VG-CPOD方案/03/file_manifest.json')
    stages = [s for s in read(ROOT / 'provenance/stages.json') if s.get('id') != '22']
    stages.append({'id':'22','line':'VG-CPOD方案','number':'03','version':'v1','title':'03 · '+TITLE,
        'path':'VG-CPOD方案/03','primary_package':PACKAGE,
        'status':'3次B组训练完成；12份评价与预测、6份权重及同seed数据分析已归档',
        'completed_runs':3,'model_epochs':30,'evaluations':12,'checkpoints':6,'predictions':12,
        'completed_cases':['B1','B2','B3'],'completed_seeds':[3351]})
    write(ROOT / 'provenance/stages.json',stages)
    owned.update(['README.md','VG-CPOD方案/README.md','tools/reproduce.py',
                  'tools/register_vgcpod_round_b.py','provenance/stages.json'])
    current = {r['path']:r for r in read(ROOT / 'provenance/current_manifest.json')}
    for rel in sorted(owned):
        path = ROOT / rel
        current[rel] = {'path':rel,'sha256':sha(path),'bytes':path.stat().st_size}
    write(ROOT / 'provenance/current_manifest.json',[current[p] for p in sorted(current)])


def check():
    entries = read(STAGE / 'file_manifest.json')
    for row in entries:
        path = STAGE / row['path']
        assert path.is_file() and sha(path) == row['sha256'],row['path']
    mapping = read(STAGE / 'package_members.json')['members']
    release = read(STAGE / mapping['release_manifest.json'])['files_sha256']
    assert set(mapping) == set(release) | {'release_manifest.json'}
    for name,digest in release.items():
        assert sha(STAGE / mapping[name]) == digest,name
    with zipfile.ZipFile(STAGE / 'results/archives' / ZIP) as archive:
        names = {n for n in archive.namelist() if not n.endswith('/')}
        assert names == {PACKAGE+'/'+n for n in mapping}
        for name,target in mapping.items():
            assert archive.read(PACKAGE+'/'+name) == (STAGE / target).read_bytes(),name
    for archive in (ZIP,TAR):
        path = STAGE / 'results/archives' / archive
        assert sha(path) == path.with_name(archive+'.sha256').read_text().split()[0]
    for row in read(STAGE / 'results/raw_export_members.json'):
        assert sha(STAGE / row['path']) == row['sha256'],row['path']
    raw = STAGE / 'results/raw_export/roundB_20261019'
    runs = list(raw.glob('B*/ut-zap50k/V*/seed*/metrics.json'))
    assert len(runs) == 3 and len(list(raw.rglob('predictions/*.npz'))) == 12
    assert len(list(raw.rglob('checkpoints/*.pt'))) == 6
    assert (raw/'launcher.exitcode').read_text().strip() == '0'
    for path in runs:
        metrics,protocol,history = read(path),read(path.parent/'protocol.json'),read(path.parent/'history.json')
        assert len(history)==10 and history[-1]['step']==3600
        assert metrics['selected_epoch']==max(range(10),key=lambda i:history[i]['val_cw']['AUC_native'])+1
        assert sha(path.parent/'checkpoints/best.pt')==metrics['checkpoint_sha256']
        assert protocol['config']==read(STAGE/'configs/round_b'/(protocol['config']['experiment_id']+'.json'))
        for name,digest in protocol['source_sha256'].items():
            assert sha(STAGE/mapping[name])==digest
    broken=[]
    docs=list(STAGE.rglob('README.md'))
    assert len(docs)==5
    for path in docs:
        text=path.read_text(encoding='utf-8')
        assert text.count('```')%2==0,path
        for link in re.findall(r'\]\(([^)\n]+)\)',text):
            link=link.split('#')[0]
            if link and not re.match(r'^[a-z]+://',link) and not (path.parent/link).exists():
                broken.append((path.relative_to(STAGE).as_posix(),link))
    assert not broken,broken
    for path in list((STAGE/'code').rglob('*.py'))+[STAGE/'analysis/recompute.py']:
        ast.parse(path.read_text(encoding='utf-8-sig'))
    return {'passed':True,'files_verified':len(entries),'package_members':len(mapping),
            'trained_runs':3,'evaluations':12,'predictions':12,'checkpoints':6,'readmes':5,'broken_links':broken}


def materialize(destination):
    destination=Path(destination).expanduser()
    if not destination.is_absolute():
        raise ValueError('Use an absolute destination')
    destination=destination.resolve()
    if destination.exists() or destination==ROOT or ROOT in destination.parents:
        raise FileExistsError(destination)
    mapping=read(STAGE/'package_members.json')['members']
    for name,source in sorted(mapping.items()):
        target=destination/PACKAGE/name
        copy(STAGE/source,target)
        if target.suffix=='.sh':
            target.chmod(target.stat().st_mode|0o111)
    release=read(destination/PACKAGE/'release_manifest.json')['files_sha256']
    assert all(sha(destination/PACKAGE/name)==digest for name,digest in release.items())
    write(destination/'materialization.json',{'line':'VG-CPOD方案','stage':'03',
        'package':PACKAGE,'files':len(mapping),'training_started':False})
    return {'line':'VG-CPOD方案','stage':'03','package':str(destination/PACKAGE),
            'files_verified':len(mapping),'training_started':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--import-data',action='store_true')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--materialize',type=Path)
    args=parser.parse_args()
    if args.import_data and args.source:
        import_data(args.source)
        result=check()
    elif args.finalize:
        finalize()
        result=check()
    elif args.check:
        result=check()
    elif args.materialize:
        result=materialize(args.materialize)
    else:
        parser.error('Choose --source PATH --import-data, --check, --finalize, or --materialize PATH')
    print(json.dumps(result,ensure_ascii=False))
