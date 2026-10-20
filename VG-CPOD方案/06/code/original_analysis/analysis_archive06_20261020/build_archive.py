"""Adapt existing E analysis prose and update the two repository indexes."""
from pathlib import Path
import re

WORK = Path(__file__).resolve().parents[1]
REPO = WORK.parent/'CZSL'
STAGE = REPO/'VG-CPOD方案/06'

text = (WORK/'VG-CPOD_v1_E组结果分析与下一轮方向_20261020.md').read_text(encoding='utf-8')
text = text.replace('# VG-CPOD v1：E 组结果分析与下一轮方向', '# 06 · 数据分析',1)
intro = '日期：2026-10-20。框架：冻结 CLIP ViT-L/14，UT-Zappos。新增运行与原导出见[结果索引](../results/README.md)，公式与修订见[主README](../README.md)。本页报告 E1 三 seed 结果，再记录 F1 的下一轮建议。'
text = re.sub(r'日期：[^\n]+',lambda _:intro,text,count=1)
text = re.sub(r'^\$\$\n(.*?)\n\$\$',lambda m:'```math\n'+m[1]+'\n```',text,flags=re.M|re.S)
text = text.replace('analysis_roundE_results_20261020/','confirmed/')
text = text.replace('(confirmed/analyze_e.py)','(../code/original_analysis/analysis_roundE_results_20261020/analyze_e.py)')
text = text.replace('(confirmed/build_report.py)','(../code/original_analysis/analysis_roundE_results_20261020/build_report.py)')
text += '''\n## 9. 仓库复算与归档核查

- [仓库复算入口](recompute.py)：读取本阶段 E1、02 的 A5／A6、04／05 的 C1／C2／D1 和 01 缓存元数据，数值计算沿用原分析。
- [仓库验证记录](repository_validation.json)：核对原包成员、权重与预测数量、README 数值、公式、链接及恢复目录。
- [原始结构化分析](confirmed/analysis.json)、[原始复算审计](confirmed/audit.json)、[分析图表](confirmed/figures/)。

```bash
python VG-CPOD方案/06/analysis/recompute.py
python tools/reproduce.py verify
```

复算输出至 `analysis/recomputed/`。原始分析和服务器指标保留在本阶段目录。F1 为后续实验计划，本轮新增训练为 E1 三个 seed、β=0.1。
'''
(STAGE/'analysis/README.md').write_text(text,encoding='utf-8')

p = REPO/'VG-CPOD方案/README.md'
text = p.read_text(encoding='utf-8')
row = '| 06 | 中心化 CP 强度扩展与双准则选权重（E 组）；E1 三 seed、两套最优权重及六项均值评估 | [修订说明](06/README.md) · [代码](06/code/README.md) · [配置](06/configs/README.md) · [结果](06/results/README.md) · [数据分析](06/analysis/README.md) |'
if row not in text:
    text = text.rstrip()+'\n'+row+'\n'
p.write_bytes(text.encode('utf-8'))

p = REPO/'README.md'
text = p.read_text(encoding='utf-8')
new = '最新归档（2026-10-20）：[VG-CPOD 06 · 中心化 CP 强度扩展与双准则选权重](VG-CPOD方案/06/README.md)，包含 E1（ρ=0.15）三 seed、native／exact 双准则选权重、六项动态／TRAIN 均值替换评估，共 72 份预测、12 份 checkpoint 文件；附公式、代码修订、超参数策略和[数据分析](VG-CPOD方案/06/analysis/README.md)。前序 [05 D组及三seed补充](VG-CPOD方案/05/README.md)、[04 C组](VG-CPOD方案/04/README.md)、[03 B组](VG-CPOD方案/03/README.md)、[02 A组](VG-CPOD方案/02/README.md)与[01 原正式结果](VG-CPOD方案/v1/results/ut-zap50k-3seeds-20261018/README.md)继续保留。'
text = re.sub(r'^最新归档[^\n]+',lambda _:new,text,flags=re.M)
text = text.replace('| VG-CPOD | 01–05 |','| VG-CPOD | 01–06 |')
text = text.replace('02–05 使用对应编号','02–06 使用对应编号').replace('02–05 可直接复用','02–06 可直接复用')
text = text.replace('05 恢复原 D 包、C 源码和三 seed 补充轮启动器。','05 恢复原 D 包、C 源码和三 seed 补充轮启动器；06 恢复原 E 包，包含双准则选权重和均值评估入口。')
p.write_text(text,encoding='utf-8')
print('Five parallel READMEs and both indexes prepared.')
