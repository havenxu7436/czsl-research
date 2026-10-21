from pathlib import Path

base=Path(__file__).resolve().parent
p=base/'build_report.py'
s=p.read_text(encoding='utf-8')
begin=s.index('## 5. 可行大路径的取舍')
end=s.index('## 7. 本次复核与交付',begin)
s=s[:begin]+'__RESEARCH_DIRECTION__\n\n'+s[end:]
s=s.replace('**建议结束现有冻结缓存头上的门控与学习率搜索，下一阶段集中做“CLIP 内部可训练视觉适配＋多层 patch 的物体条件属性学习”。G0 保留为静态对照，D1 保留为历史性能参照。** 下一阶段安排完整的结构比较和跨数据集训练，预算用于改变视觉证据的学习方式。',
'''**建议结束现有冻结缓存头上的门控与学习率搜索。下一阶段集中研究“物体条件的局部判别证据”：先在同一 patch 比较两个属性假设，再汇聚证据形成组合分数。** CLIP 视觉 Adapter 和静态 CP 作为共同基座；创新集中在判别算子及其可证明性质。G0 保留为静态对照，D1 保留为历史性能参照。正式计划为两数据集、三种读取、三 seed，共 18 次完整训练。''')
s=s.replace('**下一阶段需要同时改善材质／状态的局部辨识和物体上下文。**',
'''**下一阶段把主要投入放在可训练的局部属性判别，并继续衡量跨物体竞争。**''')
s=s.replace("    for key,value in [('RESULTS',results)",
"    text=text.replace('__RESEARCH_DIRECTION__',(OUT/'research_direction.md').read_text(encoding='utf-8'))\n    for key,value in [('RESULTS',results)")
s=s.replace('本报告中的新结构是基于这些结果和既有 CLIP 工作提出的研究决策，收益由下一阶段正式训练测量。',
'本报告给出结果判断、近邻文献差异、核心算子推导与正式比较矩阵。训练代码未改动；下一次实现以第 6 节为规格。')
p.write_text(s,encoding='utf-8',newline='\n')
print('Updated report generator')
