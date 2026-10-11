"""Build and verify a retrospective archive; never execute experimental code.

Python 3.7+, standard library only. Original files and scientific values are not
rewritten. Generated navigation copies replace links, not narrative or numbers.
"""
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
DESKTOP = Path('C:/Users/llenovo/Desktop')
EXPERIMENTS = DESKTOP / 'czsl_experiment'
PROVENANCE = ROOT / 'provenance'
LFS_EXT = {'.zip', '.gz', '.npy', '.npz', '.dat', '.pt', '.pth', '.pkl', '.jpg', '.jpeg', '.png', '.pdf'}
SKIP_DIRS = {'.git', 'tmp', '__pycache__', 'node_modules', '.venv', 'venv', 'math_render_deps'}

# Ordering is dependency ordering supported by the existing retrospective.
# Each tuple contains title, status, motivation/change, finding/next decision, sources.
STAGES = [
('原方案、三数据集基线与知识重建','已归档历史实验',
 '以 CPF 为视觉底座，恢复 OI/OE/AE、名称、GloVe 与模板组合资产；先建立基线和输入身份。',
 '历史 TEST 与后续 VAL 分开；知识重建不等于恢复原作者遗失资产。',
 ['archive_inputs/referenced_sources','archive_inputs/desktop_experiments/20261004归档','archive_inputs/desktop_experiments/CPF_all_results（20261005）','output/week5_evidence_20261005_verified/tables/historical_baseline_test_reference.csv']),
('早期 MVP、跨数据集验证与固定 prior 校准','已训练＋无训练后处理',
 'UT A/C/D/E 与 MIT、C-GQA A/C；固定 A/C checkpoint 扫 λ，以分离文本和静态 prior。',
 'UT 文本有信号；后处理选择 λ=0。20 次评价不计20次新训练。',
 ['cpf_stage_report_tools_20261004_v1','cpf_stage_report_tools_20261004_v2','output/week5_evidence_20261005','output/week5_evidence_20261005_verified','第四步_UT结果冻结_K校准诊断与跨数据集验证.md','第四步补充_lambda0后扩展AC操作.md']),
('UT v1 三 seed 语义归因与工程修复','已执行历史归因；保留缺项边界',
 '补 A_raw/A_unit/C_text/C_shuffle，检查文本收益是否需要正确对应；修复审计兼容和运行故障。',
 '技术修复不改变研究 gate；v1 未保存的绝对指标不从 v2 反推。',
 ['ekr_cpf_transition_20261005_v1','ekr_checkpoint_path_fix_v1','semantic_gate_audit_20261006_v1','semantic_gate_audit_20261006_v2','第五步_文本收益归因到EKR-CPF_完整操作方案.md','第六步_归因门槛未通过_复核与EKR决策.md','归因复核输出解读与v2修复_20261006.md','磁盘空间不足_UT归因实验恢复_20261006.md']),
('TRAIN 描述审核、UT v2 与逐图分析','已训练＋诊断',
 '一次 TRAIN 审核修订后，四组×三 seed 从初始化训练；随后导出既有预测与回放。',
 'C_text−A_unit 平均 CW +3.0670pp，C_text−shuffle CW +0.3789pp，未过预定0.5pp；OW信号保留。',
 ['ut_semantic_v2_20261006','ut_train_review_20261006_v1','ut_train_review_20261006_v2','ut_train_review_20261006_v3','output/ut_description_review_20261006','output/ut_v2_results_audit_20261007','output/ut_v2_prediction_analysis_20261007','output/ut_v2_formula_revision_20261007','archive_inputs/desktop_experiments/UT_description_review']),
('名称锚点 EKR','已训练：seed0、三组各3轮',
 '将零增量回退锚点改为 A_unit，避免关闭动态分支仍保留静态描述混淆。',
 '正确描述未胜名称和错配；先诊断机制，再决定是否扩跑。',
 ['name_anchor_ekr_20261007','output/name_anchor_ekr_20261007','output/name_ekr_pilot_review_20261007']),
('EKR 机制诊断','既有权重回放；0新训练',
 '检查 value、null、对象上下文、置换与margin；保护名称基底并确认退化原因。',
 '弱对象依赖和null拒识问题促成共享value对齐；诊断不能直接证明知识泛化。',
 ['name_ekr_mechanism_diag_20261007_v1','output/name_ekr_mechanism_diag_20261007_v1','output/name_ekr_mechanism_review_20261007']),
('共享值对齐 EKR v2：3轮','已训练：4组各3轮',
 '建立同容量名称/正确/错配对照与固定value消融，修正槽表示和共享value映射。',
 '对齐主组选择epoch0；固定value微增。后续10轮属于这些run的续训。',
 ['ekr_v2_anchor_alignment_20261007','output/ekr_v2_anchor_alignment_20261007','output/ekr_v2_review_20261008']),
('TRAIN 支持探针、SG-EKR与视觉原型诊断','支持诊断＋3组3轮；VP仅方案',
 '双半折 TRAIN 证据筛支持槽，固定value；另检查排除查询组合的视觉原型。',
 'SG支持描述未优于控制；原型仅开发诊断，不填写为正式VP训练成绩。',
 ['ekr_train_support_probe_20261008','sgekr_support_mask_20261008','output/ekr_train_support_review_20261008','output/sgekr_support_mask_20261008','output/sgekr_review_20261008']),
('EKR v2 训练预算回溯：3→10轮','续训；不是4个新run',
 '保持模型/optimizer/RNG和资产，四组从epoch3续至10，检验短训是否漏掉延迟收益。',
 '名称组epoch6改善；正确描述仍选epoch0。SG没有10轮结果。',
 ['ekr_v2_budget10_20261008','output/ekr_v2_budget10_20261008','output/ekr_v2_budget10_review_20261008']),
('CPF＋LogiCZSL B/F直接结合','已训练：UT3351、四组10轮',
 '补A_ref/B/E_ref/F，确保逻辑损失实际加入反传，冻结β=0.01、q=3。',
 '当前直接结合未产生稳定配对正增量；逻辑方向以实际服务器源码为准。',
 ['logiczsl_cpf_bf_20261008','output/BF_server_results_20261008','output/LogiCZSL论文方法重建_CPF-BF训练完整手册_20261008.md']),
('B/F梯度复核与条件属性 StageA','诊断/缓存/丢弃式检查；非正式pilot',
 '检查真实TRAIN图像的逻辑梯度，构建C_text缓存并检查条件head接线。',
 '弱且部分冲突的逻辑信号促成直接条件目标；StageA自检步数不当正式训练。',
 ['bf_conditional_preflight_20261009','conditional_attribute_stageA_20261009','output/BF_conditional_preflight_20261009','output/conditional_attribute_stageA_results_20261009','output/BF审计与条件属性StageA实践命令_20261009.md']),
('条件属性 pilot r1→r2与固定混合回放','r1工程失败；r2五组10轮＋回放',
 '改为GT-object条件属性目标，保持对象边缘；修复Torch1.13安全checkpoint兼容，另做固定25%混合回放。',
 '已见拟合与unseen迁移分开；固定读出回放不是新训练；保留失败r1。',
 ['conditional_attribute_pilot_20261009','pilot_review_20261009','output/conditional_attribute_pilot_results_20261009_r2']),
('Generic同结构知识对照＋25%混合','三项新训练＋G_K复用',
 '控制容量和读出，只改名称、正确和两置换内容，区分工程增益与正确知识贡献。',
 '四组CW均提升，内容归因未成立；不能把复用G_K列为第四项新训练。',
 ['generic_mixture_controls_20261009','generic_controls_review_20261010','output/generic_mixture_controls_results_20261010']),
('冻结 CLIP 探针 v1→r2','v1检查失败；r2七配方、0optimizer',
 '修正广义seen/unseen划分assert，检查图像隔离；以冻结CLIP判断知识教师信号。',
 'OE+AE与AE-only继续gate均False；七种配方不算七个训练模型。',
 ['clip_knowledge_probe_20261010','clip_knowledge_probe_20261010_r2','clip_probe_review_20261010','output/clip_knowledge_probe_results_20261010_r2']),
('TRAIN逐句审查、知识来源与研究回溯','36逐句视图＋图片审查；0新训练',
 '检查均值是否隐藏局部信号，保留研究目标与实现完成度的解释修正。',
 'AE第三条有局部信号但净收益集中Leather；不把TRAIN审查当unseen泛化证据。',
 ['train_knowledge_audit_20261010','train_knowledge_audit_review_20261010','output/train_knowledge_audit_results_20261010','experiment_history_20261010','knowledge_history_20261010']),
('residual-0601：统一A–H与sanity','已完成12组×10轮、48份评价',
 '以旧native CPF＋residual接口补齐八主组与四独立对照，统一VAL选模与VAL/TEST评价。',
 '核心工程范围完成，完整H低于A；先验、文本、逻辑与head影响按配对和边界解释。',
 ['reproduce_0601_20261010','review_0601_20261011','output/0601_priority_results_20261011','output/0601_result_review_20261011','output/0601结果分析与下一步实践命令_20261011.md']),
('formula-aligned-0601：逐公式修正r2','代码/交付/CPU验证；未收集正式训练结果',
 '按公式审计修正G/H原始head评分、GT-object CE、逻辑grounding、prior及train/infer共用接口。',
 '保留接口选择与未覆盖项；42次×30轮为新训练计划，不继承residual旧表。',
 ['faithful_0601_20261011','review_0601_formula_20261011','desktop_bundle_0601_20261011','archive_inputs/formula_aligned_bundle','output/0601逐公式复刻审计_20261011.md','output/0601主公式复刻_实践命令与时间预估_20261011.md']),
('residual描述删除扩展与后续候选','仅代码交付；本地未收集正式结果',
 '保留H_noise10/30/50追加协议和代码，明确描述槽删除范围及与主公式对齐版的不同。',
 '没有对应complete/metrics证据前不标成已训练；其他共享视觉/教师/多seed方案也保持待执行。',
 ['extend_0601_20261011','output/0601_noise_extension_release_20261011.zip']),
]

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def skip(path, base):
    parts=path.relative_to(base).parts
    return any(p in SKIP_DIRS or p.endswith('_deps') for p in parts) or path.suffix in {'.pyc','.pyo'}

def copy_identity(source, target, mapping):
    target.parent.mkdir(parents=True, exist_ok=True)
    identity=digest(source)
    if target.exists():
        if digest(target)!=identity:
            raise RuntimeError('Preserve conflicting archive destination: '+str(target))
    else:
        shutil.copy2(str(source), str(target))
    mapping[source.resolve().as_posix().lower()]={
        'original_path':source.resolve().as_posix(),
        'repository_path':target.relative_to(ROOT).as_posix(),
        'bytes':source.stat().st_size, 'sha256':identity}

def external_reason(path):
    rel=path.relative_to(EXPERIMENTS).as_posix()
    low=rel.lower()
    if skip(path, EXPERIMENTS): return '临时缓存/依赖/bytecode'
    if low.startswith('czsl_paper/'): return '论文阅读资料；非实验结果（被来源清单引用的另行保存）'
    if '/downloads/' in low or '/pretrained/' in low: return '数据集/词向量下载或预训练资源'
    if path.stat().st_size>50*2**20:
        if path.suffix in {'.pt','.pth','.ckpt'}: return '大型训练/预训练权重；原存储位置保留'
        if low.endswith(('.tar.gz','.tgz')): return '全量备份容器；已解包科学资料分别收录'
        if path.suffix=='.zip':
            with zipfile.ZipFile(str(path)) as archive:
                if any(x.file_size>50*2**20 and (x.filename.lower().endswith(('.pt','.pth','.ckpt')) or '/downloads/' in x.filename.lower()) for x in archive.infolist()):
                    return '含大型权重/数据的备份容器；小型科学资料分别收录'
        elif path.suffix not in LFS_EXT:
            return '外部大资源；保留位置及大小，不并入研究结果'
    return None

def resolve_old(raw):
    p=Path(raw)
    if p.is_file(): return p
    normalized=raw.replace('\\','/').lstrip('/')
    aliases={
        (DESKTOP/'CPF_all_results').as_posix():(EXPERIMENTS/'CPF_all_results（20261005）').as_posix(),
        (DESKTOP/'UT_description_review').as_posix():(EXPERIMENTS/'UT_description_review').as_posix(),
    }
    for old,new in aliases.items():
        if normalized.lower().startswith(old.lower()+'/'):
            candidate=Path(new+normalized[len(old):])
            if candidate.is_file(): return candidate
    return None

def import_sources():
    mapping={}
    exclusions=[]
    for source in sorted(EXPERIMENTS.rglob('*')):
        if not source.is_file(): continue
        reason=external_reason(source)
        if reason:
            exclusions.append({'original_path':source.as_posix(),'bytes':source.stat().st_size,'reason':reason})
            continue
        copy_identity(source, ROOT/'archive_inputs/desktop_experiments'/source.relative_to(EXPERIMENTS),mapping)
    bundle=DESKTOP/'0601原设计复刻_代码与配置_20261011'
    for source in sorted(bundle.rglob('*')):
        if source.is_file() and not skip(source,bundle):
            copy_identity(source,ROOT/'archive_inputs/formula_aligned_bundle'/source.relative_to(bundle),mapping)
    # These two receipts enumerate the exact external inputs of the existing reports.
    for receipt in sorted((ROOT/'output').glob('*.sources.json')):
        obj=json.loads(receipt.read_text(encoding='utf-8-sig'))
        for key,value in obj.get('source_files',{}).items():
            raw=value.get('path',key)
            source=resolve_old(raw)
            if source is None:
                continue
            try:
                source.relative_to(ROOT)
                continue
            except ValueError:
                pass
            if '/downloads/' in source.as_posix().lower() and source.stat().st_size>50*2**20:
                exclusions.append({'original_path':source.as_posix(),'bytes':source.stat().st_size,'reason':'来源引用的大型词向量/数据资源；记录来源，原文件保留'})
                continue
            if source.resolve().as_posix().lower() not in mapping:
                target=ROOT/'archive_inputs/referenced_sources'/source.relative_to(DESKTOP)
                copy_identity(source,target,mapping)
    write_json(PROVENANCE/'external_sources.json', list(mapping.values()))
    write_csv(PROVENANCE/'external_excluded.csv',exclusions,['original_path','bytes','reason'])
    return mapping

def portable_report(source, target, mapping):
    text=source.read_text(encoding='utf-8-sig')
    def replace(match):
        label, raw=match.group(1),match.group(2)
        raw=unquote(raw.strip().strip('<>')).replace('\\','/')
        original=raw.lstrip('/') if raw.startswith('/C:') else raw
        if not re.match(r'^[A-Za-z]:/',original): return match.group(0)
        found=resolve_old(original)
        destination=None
        if found:
            try:
                destination=found.relative_to(ROOT)
            except ValueError:
                item=mapping.get(found.resolve().as_posix().lower())
                if item: destination=Path(item['repository_path'])
        if destination is None:
            return label+'（原存储位置：`'+original+'`）'
        relative=os.path.relpath(str(ROOT/destination),str(target.parent)).replace('\\','/')
        return '['+label+'](<'+relative+'>)'
    text=re.sub(r'\[([^\]\n]+)\]\((<[^>]+>|[^)\n]+)\)',replace,text)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text,encoding='utf-8')

def stage_of(path):
    for i,stage in reversed(list(enumerate(STAGES,1))):
        for prefix in stage[4]:
            if path==prefix or path.startswith(prefix+'/'):
                return '{:02d}'.format(i)
    if path.startswith('docs/stages/'): return path.split('/')[2][:2]
    return '19'

def build_docs(mapping):
    registry=[]
    text=['# 研究阶段索引','','按研究依赖顺序回溯整理；同日工作可能交叉。原始证据和现有总档案支持此顺序，未伪造实验完成时刻。','','| 顺序 | 阶段 | 实际状态 |','|---|---|---|']
    for i,(title,status,change,finding,sources) in enumerate(STAGES,1):
        identity='{:02d}'.format(i)
        card=ROOT/'docs/stages'/('{}_stage.md'.format(identity))
        lines=['# {}. {}'.format(identity,title),'','状态：'+status+'。','','## 修改动机与实际工作','',change,'','## 结果与决策','',finding,'','## 原始证据入口','']
        existing=[]
        for relative in sources:
            p=ROOT/relative
            if p.exists():
                lines.append('- [{}](<{}>)'.format(relative,os.path.relpath(str(p),str(card.parent)).replace('\\','/')))
                existing.append(relative)
        lines+=['','完整公式、对照、全部数值及比较限制请结合 [实验总档案](../history_20261010.md) 与 [路线对照](../ROUTES.md) 查阅；第16—18阶段的最新状态以各自结果/交付文件为准。','']
        card.parent.mkdir(parents=True,exist_ok=True)
        card.write_text('\n'.join(lines),encoding='utf-8')
        text.append('| {} | [{}](stages/{}_stage.md) | {} |'.format(identity,title,identity,status))
        registry.append({'stage':identity,'title':title,'status':status,'change':change,'finding':finding,'sources':existing})
    text+=['','历史总档案包含更细的24项执行/失败/回放事件和公式修正总表；这里将有直接依赖的事件归到18张研究阶段卡。不能将阶段数当作独立训练次数。','']
    (ROOT/'docs/EXPERIMENTS.md').write_text('\n'.join(text),encoding='utf-8')
    write_json(PROVENANCE/'stages.json',registry)
    for src,name in [
        ('output/截至20261010_CZSL全部实验_公式修正_决策与数据分析总档案_排版复核版_r3.md','history_20261010.md'),
        ('output/截至20261010_知识恢复构建处理_来源与作用完整档案.md','knowledge_history_20261010.md'),
        ('output/0601结果分析与下一步实践命令_20261011.md','residual_review_20261011.md')]:
        portable_report(ROOT/src,ROOT/'docs'/name,mapping)

def metric_index():
    rows=[]
    schemas=[('02','output/week5_evidence_20261005_verified/tables/all_val_metrics.csv','auc_pct','hm_pct','dataset','seed','variant','phase','setting','best_epoch'),
        ('04','archive_inputs/desktop_experiments/UT_description_review/UT_v2_results_20261007_153654/files/experiments/semantic_v2/reports/paired_v2/runs.csv','auc_pct','hm_pct','dataset','seed','variant','phase','setting','best_epoch'),
        ('05','output/name_ekr_pilot_review_20261007/final_metrics.csv','auc_pct','hm_pct',None,None,'variant',None,'setting','best_epoch'),
        ('07','output/ekr_v2_review_20261008/artifacts/reports/results.csv','auc_pct','hm_pct',None,'seed','variant',None,'setting','selected_epoch'),
        ('08','output/sgekr_review_20261008/artifacts/reports/results.csv','auc_pct','hm_pct',None,'seed','variant',None,'setting','selected_epoch'),
        ('09','output/ekr_v2_budget10_review_20261008/artifacts/reports/results.csv','auc_pct','hm_pct',None,'seed','variant',None,'setting','selected_epoch'),
        ('10','output/BF_server_results_20261008/BF_seed3351_实际训练与结果.csv','AUC','best_hm',None,'seed','variant',None,'setting','best_epoch'),
        ('16','output/0601_priority_results_20261011/extracted/summary/metrics.csv','AUC_pct','best_hm_pct',None,None,'variant','phase','setting','best_epoch')]
    for st,rel,auc,hm,dataset,seed,variant,phase,setting,epoch in schemas:
        p=ROOT/rel
        if not p.is_file(): raise RuntimeError('Missing required metric source: '+rel)
        for n,item in enumerate(csv.DictReader(io.StringIO(p.read_text(encoding='utf-8-sig'))),2):
            rows.append({'stage':st,'route':'residual-0601' if st=='16' else 'research-history','dataset':item.get(dataset,'ut-zappos') if dataset else 'ut-zappos','seed':item.get(seed,'') if seed else ('3351' if st=='16' else '0'),'variant':item[variant],'phase':item.get(phase,'val') if phase else 'val','setting':item[setting],'selected_epoch':item.get(epoch,''),'auc_pct':item[auc],'hm_pct':item[hm],'source':rel,'source_csv_line':n,'note':'historical identity; not an additional independent run'})
    for st,rel in [('12','output/conditional_attribute_pilot_results_20261009_r2/extracted/pilot/summary.json'),('13','output/generic_mixture_controls_results_20261010/extracted/controls/summary.json')]:
        obj=json.loads((ROOT/rel).read_text(encoding='utf-8-sig'))
        for variant,value in obj['results'].items():
            selected=value['selected']
            for setting,metric in selected['metrics_raw'].items():
                rows.append({'stage':st,'route':'research-history','dataset':'ut-zappos','seed':'0','variant':variant,'phase':'val','setting':setting,'selected_epoch':selected['epoch'],'auc_pct':float(metric['AUC'])*100,'hm_pct':float(metric['best_hm'])*100,'source':rel,'source_csv_line':'','note':'JSON raw fractions multiplied by 100; G_K reused prior run' if variant=='G_K' else 'JSON raw fractions multiplied by 100'})
    fields=['stage','route','dataset','seed','variant','phase','setting','selected_epoch','auc_pct','hm_pct','source','source_csv_line','note']
    write_csv(PROVENANCE/'metrics_index.csv',rows,fields)
    groups=defaultdict(list)
    for row in rows: groups[row['stage']].append(row)
    lines=['# 指标索引','','此页是来源索引，完整机器表见 [metrics_index.csv](../provenance/metrics_index.csv)。AUC/HM单位为百分数。不同数据集、seed、协议、预算和评价split不跨阶段排名。JSON原始比例乘100；CSV百分数保持原值。','','| 阶段 | 评价记录行数 | 原始来源 |','|---|---:|---|']
    for st,items in sorted(groups.items()):
        refs=sorted({x['source'] for x in items})
        lines.append('| {} | {} | {} |'.format(st,len(items),'、'.join('[原始文件](<../'+ref+'>)' for ref in refs)))
    lines+=['','CLIP探针、TRAIN逐句、固定λ与固定混合回放使用不同诊断指标，见完整总档案及其原始JSON，未填入正式训练AUC表。v1缺失指标不补造。主公式对齐版暂无正式结果，因而指标表不出现它的研究成绩。','']
    (ROOT/'docs/METRICS.md').write_text('\n'.join(lines),encoding='utf-8')

def files_for_archive():
    listed=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=str(ROOT))
    return sorted(ROOT/x.decode('utf-8') for x in listed.split(b'\0') if x)

def validate_links():
    failures=[]
    checked=0
    for p in [ROOT/'README.md']+list((ROOT/'docs').rglob('*.md')):
        for raw in re.findall(r'\]\((<[^>]+>|[^)\n]+)\)',p.read_text(encoding='utf-8')):
            raw=unquote(raw.strip('<>')).split('#')[0]
            if not raw or re.match(r'^[a-z]+://',raw): continue
            checked+=1
            if not (p.parent/raw).exists(): failures.append({'document':p.relative_to(ROOT).as_posix(),'target':raw})
    return checked,failures

def manifest():
    records=[]
    secrets=[]
    excluded={'provenance/file_manifest.csv','provenance/archive_validation.json','provenance/commit_plan.json'}
    token=re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    for p in files_for_archive():
        rel=p.relative_to(ROOT).as_posix()
        if rel in excluded: continue
        size=p.stat().st_size
        if size>100*2**20 and p.suffix not in LFS_EXT: raise RuntimeError('Unmanaged oversized file: '+rel)
        if p.suffix in {'.md','.py','.ps1','.json','.yml','.yaml','.txt','.csv','.sh'} and size<8*2**20:
            for n,line in enumerate(p.read_text(encoding='utf-8',errors='replace').splitlines(),1):
                if token.search(line): secrets.append({'path':rel,'line':n})
        records.append({'path':rel,'bytes':size,'sha256':digest(p),'storage':'git-lfs' if p.suffix in LFS_EXT else 'git','stage':stage_of(rel)})
    if secrets: raise RuntimeError('Credential-like content found; locations only: '+json.dumps(secrets))
    write_csv(PROVENANCE/'file_manifest.csv',records,['path','bytes','sha256','storage','stage'])
    checked,failures=validate_links()
    if failures: raise RuntimeError('Broken navigation links: '+json.dumps(failures,ensure_ascii=False))
    summary={'archive_date':'2026-10-11','timezone':'Asia/Shanghai','files':len(records),'bytes':sum(r['bytes'] for r in records),'lfs_bytes':sum(r['bytes'] for r in records if r['storage']=='git-lfs'),'stage_cards':len(STAGES),'navigation_links_checked':checked,'broken_links':failures,'credential_patterns_found':0,'original_files_rewritten':False,'new_training_started':False,'large_external_resources':'See external_excluded.csv; originals retained','historical_commits_are_retrospective':True}
    write_json(PROVENANCE/'archive_validation.json',summary)
    print(json.dumps(summary,ensure_ascii=False))

def verify():
    errors=[]
    records=list(csv.DictReader((PROVENANCE/'file_manifest.csv').open(encoding='utf-8-sig')))
    for item in records:
        p=ROOT/item['path']
        if not p.is_file() or p.stat().st_size!=int(item['bytes']) or digest(p)!=item['sha256']:
            errors.append(item['path'])
    checked,links=validate_links()
    print(json.dumps({'passed':not errors and not links,'manifest_files_checked':len(records),'changed_or_missing':errors,'navigation_links_checked':checked,'broken_links':links,'training_started':False},ensure_ascii=False))
    if errors or links: sys.exit(1)

def commit():
    resumed=subprocess.run(['git','rev-parse','--verify','HEAD'],cwd=str(ROOT),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
    plan=[]
    if resumed:
        # Only resume our exact consecutive retrospective import, never arbitrary history.
        history=subprocess.check_output(['git','log','--reverse','--format=%H%x00%s'],cwd=str(ROOT)).decode('utf-8').splitlines()
        expected=['Initialize private research navigation and provenance tools']+['Retrospective archive stage {:02d}: {}'.format(i+1,s[0]) for i,s in enumerate(STAGES)]
        if len(history)>len(expected):
            raise RuntimeError('Preserve already completed or unrelated history.')
        for i,line in enumerate(history):
            sha,title=line.split('\0',1)
            if title!=expected[i]:
                raise RuntimeError('Preserve unrelated history; cannot resume import.')
            changed=subprocess.check_output(['git','diff-tree','--root','--no-commit-id','--name-only','-r','-z',sha],cwd=str(ROOT))
            plan.append({'stage':'{:02d}'.format(i),'files':len([p for p in changed.split(b'\0') if p]),'commit':sha,'title':title})
        if subprocess.check_output(['git','branch','--show-current'],cwd=str(ROOT)).decode().strip()!='main':
            raise RuntimeError('Resume must preserve the existing main branch.')
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=str(ROOT)).strip():
        raise RuntimeError('Preserve pre-existing staged files.')
    subprocess.check_call(['git','branch','-M','main'],cwd=str(ROOT))
    subprocess.check_call(['git','lfs','install','--local'],cwd=str(ROOT))
    paths=[x.decode('utf-8') for x in subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=str(ROOT)).split(b'\0') if x]
    groups=defaultdict(list)
    for path in paths:
        scaffold=path in {'README.md','.gitignore','.gitattributes'} or path.startswith('archive_tools/') or (path.startswith('docs/') and not path.startswith('docs/stages/'))
        group=('19' if resumed else '00') if scaffold else stage_of(path)
        if resumed and group!='19' and int(group)<len(plan):
            raise RuntimeError('Unexpected new file in previously committed stage: '+path)
        groups[group].append(path)
    modified=[x.decode('utf-8') for x in subprocess.check_output(['git','diff','--name-only','-z'],cwd=str(ROOT)).split(b'\0') if x]
    for path in modified:
        if not (path=='README.md' or path.startswith(('docs/','archive_tools/','provenance/'))):
            raise RuntimeError('Preserve unrelated tracked modification: '+path)
        groups['19'].append(path)
    for group,files in sorted(groups.items()):
        subprocess.run(['git','add','--pathspec-from-file=-','--pathspec-file-nul'],cwd=str(ROOT),input=('\0'.join(files)+'\0').encode('utf-8'),check=True)
        title='Initialize private research navigation and provenance tools' if group=='00' else ('Retrospective archive stage '+group+': '+STAGES[int(group)-1][0] if group!='19' else 'Retrospective archive: remaining reports, releases and provenance')
        subprocess.check_call(['git','commit','--quiet','-m',title],cwd=str(ROOT))
        plan.append({'stage':group,'files':len(files),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip(),'title':title})
        print('Committed stage {}: {} files'.format(group,len(files)),flush=True)
    write_json(PROVENANCE/'commit_plan.json',plan)
    subprocess.check_call(['git','add','provenance/commit_plan.json'],cwd=str(ROOT))
    subprocess.check_call(['git','commit','--quiet','-m','Document retrospective commit identities and stage order'],cwd=str(ROOT))
    subprocess.check_call(['git','tag','-a','archive-20261011','-m','Frozen private research archive; experimental dates come from original protocols'],cwd=str(ROOT))
    print(json.dumps({'commits':len(plan)+1,'tag':'archive-20261011','files':len(paths)},ensure_ascii=False))

if __name__=='__main__':
    action=sys.argv[1] if len(sys.argv)>1 else 'verify'
    if action=='build':
        PROVENANCE.mkdir(exist_ok=True)
        mapping=import_sources()
        build_docs(mapping)
        metric_index()
        manifest()
    elif action=='verify': verify()
    elif action=='commit': commit()
    elif action=='refresh':
        # Refresh generated navigation/receipts without recopying original files.
        imported=json.loads((PROVENANCE/'external_sources.json').read_text(encoding='utf-8'))
        exclusions=list(csv.DictReader((PROVENANCE/'external_excluded.csv').open(encoding='utf-8-sig')))
        retained=[]
        for entry in imported:
            if '/downloads/' in entry['original_path'].lower() and entry['bytes']>50*2**20:
                exclusions.append({'original_path':entry['original_path'],'bytes':entry['bytes'],'reason':'来源引用的大型词向量/数据资源；记录来源，原文件保留'})
            else: retained.append(entry)
        mapping={entry['original_path'].lower():entry for entry in retained}
        write_json(PROVENANCE/'external_sources.json',retained)
        write_csv(PROVENANCE/'external_excluded.csv',exclusions,['original_path','bytes','reason'])
        build_docs(mapping)
        metric_index()
        manifest()
    else: raise SystemExit('Use build, refresh, verify or commit.')
