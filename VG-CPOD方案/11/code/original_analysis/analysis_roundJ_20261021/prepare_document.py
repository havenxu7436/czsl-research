"""Prepare document build and rendered-layout checks for J1."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
text=(ROOT/'analysis_roundI_20261021/build_document.py').read_text(encoding='utf-8')
text=text.replace('H release', 'J release').replace('roundI', 'roundJ').replace('I组', 'J组').replace('EXPERIMENT_I', 'EXPERIMENT_J')
(HERE/'build_document.py').write_text(text,encoding='utf-8',newline='\n')
text=(ROOT/'analysis_roundI_20261021/verify_document.cjs').read_text(encoding='utf-8')
text=text.replace('I组', 'J组').replace('2.1 共享 TRAIN', '2.1 固定 TRAIN')
text=text.replace('2.2 两组输入', '2.2 拆分静态').replace('2.3 全局 CP', '2.3 联合概率')
text=text.replace('2.4 权重选择', '2.4 TRAIN').replace('5.1 激活', '5.1 三张 GPU')
(HERE/'verify_document.cjs').write_text(text,encoding='utf-8',newline='\n')
