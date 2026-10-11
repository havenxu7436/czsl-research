"""Compile a retrospective knowledge-asset document; never rebuild model assets."""
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESKTOP = Path('C:/Users/llenovo/Desktop/czsl_experiment')
ARCHIVE = DESKTOP / 'CPF_all_results（20261005）/czsl_all_completed_20261005_000428_2772161/files'
DRAFT = DESKTOP / '20261004归档/knowledge/kcc_logic_3datasets_reconstructed_v1'
V1 = ROOT / 'output/week5_evidence_20261005_verified/evidence/knowledge_source'
V2 = ROOT / 'ut_semantic_v2_20261006/knowledge/ut-zap50k'
COND = ROOT / 'output/BF_conditional_preflight_20261009/extracted/conditional'
BF = ROOT / 'logiczsl_cpf_bf_20261008'
TRAIN = ROOT / 'output/train_knowledge_audit_results_20261010/extracted'
TITLE = '截至20261010_知识恢复构建处理_来源与作用完整档案'
OUTPUT = ROOT / 'output' / (TITLE + '.md')
sources = {}
checks = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def add(key, path, label, role):
    path = Path(path).resolve()
    assert path.is_file(), path
    sources[key] = dict(path=str(path), label=label, role=role, sha256=sha(path), bytes=path.stat().st_size)
    return path


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def link(key):
    item = sources[key]
    return '[' + item['label'] + '](</' + Path(item['path']).as_posix() + '>)'


def table(headers, rows):
    def cell(v):
        return str(v).replace('|', '\\|').replace('\n', '<br>')
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows])


add('plan', 'C:/Users/llenovo/Desktop/0601origin/ZSCL实验设计0601.md', '最初 0601 方案', '设计来源，不是已执行事实')
add('full_report', ROOT / 'output/截至20261010_CZSL全部实验_公式修正_决策与数据分析总档案.md', '全部实验总档案', '已有实验与决策的配套报告，保持不变')
add('original_paper', 'C:/Users/llenovo/Desktop/CZSL/Wu_LOGICZSL_Exploring_Logic-induced_Representation_for_Compositional_Zero-shot_Learning_CVPR_2025_paper.pdf', 'LogiCZSL 原论文 PDF', '此前论文方法核对的原件；本次只索引与校验文件身份，不重新解析 PDF')
add('author_copy', 'C:/Users/llenovo/Desktop/CZSL/CZSL_code/LogiCZSL_copy.tar.gz', '本地作者代码副本', '此前逻辑公式/接口审计来源；不包含原作者历史知识 JSON')
add('glove_archive', 'C:/Users/llenovo/Desktop/CZSL/CZSL_code/downloads/glove.6B.zip', '本地 GloVe 6B 原始 ZIP', '可用词向量源包；实际编码成员身份以完整 glove.6B.300d.txt SHA 为准')
add('wordnet_license', DRAFT / 'WORDNET_LICENSE.txt', 'WordNet 原始许可', '初始知识生成包附带的 WordNet 许可')
for name in ('generate.py', 'build_text_prior.py', 'validate_server.py'):
    add(name, DRAFT / 'scripts' / name, name, '初期三数据集知识包的实际脚本')
for name in ('generation_manifest.json', 'build_compatibility.py', 'compatibility.csv', 'README.md',
             'oi_dict.json', 'oe_dict.json', 'ae_dict.json', 'validate_knowledge.py', 'knowledge-files.sha256'):
    add('initial_' + name, ARCHIVE / 'reconstructed_knowledge/ut-zap50k' / name, '最早 UT / ' + name, '2026-09-29 非官方语义重建及旧候选掩码')
for dataset in ('ut-zap50k', 'mit-states', 'cgqa'):
    for name in ('attributes.txt', 'objects.txt', 'oi_dict.json', 'oe_dict.json', 'ae_dict.json',
                 'entry_provenance.csv', 'review_status.csv', 'manifest.json', 'K_oa_text_lexical.csv', 'K_oa_text_report.json'):
        add('v1_' + dataset + '_' + name, V1 / dataset / name, 'v1 ' + dataset + ' / ' + name, '已核对早期 cache 输入的源知识副本')
    manifest = js(V1 / dataset / 'manifest.json')
    assert manifest['generator_sha256'] == sources['generate.py']['sha256']
    assert manifest['text_prior']['builder_sha256'] == sources['build_text_prior.py']['sha256']
    for name, digest in manifest['knowledge_sha256'].items():
        assert sha(V1 / dataset / name) == digest, (dataset, name)
    checks.append({'check': 'v1 source hashes and original generation/lexical builder identities', 'dataset': dataset, 'passed': True})
for name in ('attributes.txt', 'objects.txt', 'oi_dict.json', 'oe_dict.json', 'ae_dict.json', 'review_status.csv', 'manifest.json', 'item_review.json'):
    add('v2_' + name, V2 / name, 'UT-v2 / ' + name, 'TRAIN 图片审查后冻结的源知识')
for name in ('manifest.json', 'selected_train_images.json', 'original_descriptions.md'):
    add('gallery_' + name, ROOT / 'ut_semantic_v2_20261006/review_evidence' / name, 'TRAIN 审图 / ' + name, 'v2 知识改写的图片抽样与来源证据')
for name in ('manifest.json', 'labels.json', 'report.json', 'cache_manifest.json'):
    add('cache_' + name, COND / 'cache' / name, 'UT-v2 cache / ' + name, '服务器构建后回传的缓存元数据；此目录没有所有缓存张量')
for name in ('bank.pt', 'manifest.json', 'labels.json', 'descriptions.json', 'report.json'):
    add('bank_' + name, COND / 'bank' / name, 'UT-v2 EKR bank / ' + name, '服务器构建后回传的描述槽资产')
add('glove_builder', ROOT / 'reproduce_0601_20261010/reference_source/tools/build_knowledge.py', '冻结 GloVe / build_knowledge.py', '原缓存构建脚本的本地冻结副本')
add('aliases', ROOT / 'reproduce_0601_20261010/reference_source/models/word_embedding_utils.py', 'CPF 名称别名 / word_embedding_utils.py', '名称编码来源')
add('cache_loader', ROOT / 'reproduce_0601_20261010/reference_source/models/kcc_cache.py', '原 Text/Prior 读取 / kcc_cache.py', '实际模型消费及校验')
add('cpf_consumer', ROOT / 'reproduce_0601_20261010/reference_source/models/kcc_cpf.py', 'CPF Text/Prior 前向 / kcc_cpf.py', '固定知识表在文本组合、文字引导和分类中的具体消费')
add('clip_distribution', ROOT / 'clip_knowledge_probe_20261010_r2/metrics.py', 'CLIP-r2 / metrics.py', '图文分数沿属性轴转换为条件教师概率')
add('bank_builder', ROOT / 'ut_semantic_v2_20261006/transition/bank.py', 'EKR / bank.py', '描述槽、均值、残差、固定错配的构建')
add('ekr_router', ROOT / 'ut_semantic_v2_20261006/transition/router.py', '旧 EKR / router.py', '中心化 key/value 与对象条件查询的读取')
add('name_router', ROOT / 'name_anchor_ekr_20261007/transition/name_router.py', '名称锚点 / name_router.py', '名称控制的非零值与仅名称对象上下文')
add('aligned_values', ROOT / 'ekr_v2_anchor_alignment_20261007/core.py', '共享值对齐 / core.py', '原始槽的低秩 value 处理，不重写源知识')
add('train_support_probe', ROOT / 'ekr_train_support_probe_20261008/probe.py', '初次 TRAIN 语义支持 / probe.py', '固定语义几何的初步诊断，不是 SG 两折 mask')
add('sg_consumer', ROOT / 'sgekr_support_mask_20261008/core.py', 'SG-EKR / core.py', '固定半单位值、recipient 掩码与视觉查询')
add('v2_prepare', ROOT / 'ut_semantic_v2_20261006/transition/v2_prepare.py', 'UT-v2 / v2_prepare.py', 'v2 质量门槛与构建串联')
for key, rel, label, role in (
    ('bf_knowledge', 'assets/ut-zap50k/knowledge.json', 'B/F / knowledge.json', '另一套论文方法重建源知识'),
    ('bf_builder', 'build_bank.py', 'B/F / build_bank.py', '固定逻辑节点编码'),
    ('bf_common', 'common.py', 'B/F / common.py', 'OI/OE/AE 结构校验'),
    ('bf_adapter', 'model/bf_logic.py', 'B/F / bf_logic.py', '原生 CPF 视觉分支到知识节点的 grounding')):
    add(key, BF / rel, label, role)
add('bf_loss', ROOT / 'reproduce_0601_20261010/reference_source/models/bf_logic_loss.py', 'B/F / bf_logic_loss.py', '论文三项逻辑约束的实际实现')
for name in ('manifest.json', 'bank.pt'):
    add('bf_' + name, ROOT / 'reproduce_0601_20261010/assets/logic_bank' / name, 'B/F 固定逻辑 bank / ' + name, '后续执行包携带的同源 B/F 逻辑资产')
add('bf_manual', ROOT / 'output/LogiCZSL论文方法重建_CPF-BF训练完整手册_20261008.md', 'LogiCZSL 方法重建手册', '论文核对、原代码审计与适配说明；部分状态是发布时状态')
add('sg_builder', ROOT / 'sgekr_support_mask_20261008/pilot.py', 'SG-EKR / pilot.py', 'TRAIN 视觉支持掩码构建')
add('sg_mask', ROOT / 'output/sgekr_review_20261008/artifacts/support_mask.json', 'SG-EKR / support_mask.json', '已运行的两折视觉支持选择结果')
add('conditional_head', ROOT / 'conditional_attribute_pilot_20261009/head.py', '条件属性头 / head.py', '均值描述与名称/错配控制的实际处理')
add('stageA_prepare', ROOT / 'conditional_attribute_stageA_20261009/prepare.py', 'StageA / prepare.py', '冻结 CPF 视觉/基准分数缓存，不是源知识')
add('clip_protocol', ROOT / 'clip_knowledge_probe_20261010_r2/protocol.json', 'CLIP-r2 / protocol.json', '固定模型、知识、名称、置换和数据身份')
add('clip_prompts', ROOT / 'clip_knowledge_probe_20261010_r2/prompts.py', 'CLIP-r2 / prompts.py', '七组等权九槽 prompt 构建')
add('clip_probe', ROOT / 'clip_knowledge_probe_20261010_r2/probe.py', 'CLIP-r2 / probe.py', '冻结图文编码、均值和教师概率计算')
for name in ('ae_dict.json', 'oe_dict.json'):
    add('clip_' + name, ROOT / 'clip_knowledge_probe_20261010_r2/assets' / name, 'CLIP-r2 / ' + name, '复制 v2 原文，不是新改写知识')
add('clip_result', ROOT / 'output/clip_knowledge_probe_results_20261010_r2/czsl_clip_knowledge_probe_20261010_r2.zip', 'CLIP-r2 结果 ZIP', '包含图文缓存、prompts、教师输出和审计；本报告不展开大数组')
add('clip_review', ROOT / 'output/clip_knowledge_probe_results_20261010_r2/CLIP知识证据探针_r2_独立复核与下一步判断_20261010.md', 'CLIP-r2 独立复核', '现有探针结论')
for name in ('input_reads.json', 'complete.json', 'human_review_blank.json', 'sentence_train_report.json', 'unit_text_receipt.json'):
    add('train_' + name, TRAIN / 'result' / name, 'TRAIN 审查 / ' + name, '逐句证据、输入范围与未填人工审查表')
add('train_run', TRAIN / 'source/run.py', 'TRAIN 审查 / run.py', '重编码单槽文本并分析已有 TRAIN 缓存')
add('train_manifest', TRAIN / 'source/train_input_manifest.json', 'TRAIN 审查 / train_input_manifest.json', '逐文件锁定已使用的原 CLIP-r2 输入')
add('train_review', ROOT / 'output/train_knowledge_audit_results_20261010/TRAIN知识落地审查_独立复核与路线判断_20261010.md', 'TRAIN 知识落地复核', '已完成的审查结论，不是知识新版本')
add('full_feature', ROOT / 'reproduce_0601_20261010/source/models/full_feature.py', '0601 G/H / full_feature.py', '已实现的 pair block 消费路径，尚无本报告核实的训练结果')
add('full_readme', ROOT / 'reproduce_0601_20261010/README.md', '0601 G/H 执行包说明', '后续实现/待执行与已有结果分界')
add('v1_summary', ROOT / 'output/week5_evidence_20261005_verified/tables/knowledge_summary.csv', '三数据集知识编码统计', '早期缓存数值与来源摘要')

v2_manifest = js(V2 / 'manifest.json')
for name, digest in v2_manifest['knowledge_sha256'].items():
    assert sha(V2 / name) == digest, name
cache = js(COND / 'cache/manifest.json')
assert sha(sources['glove_builder']['path']) == cache['builder_sha256']
assert sha(COND / 'cache/manifest.json') == sha(COND / 'cache/cache_manifest.json')
bank = js(COND / 'bank/manifest.json')
for name, digest in bank['files_sha256'].items():
    assert sha(COND / 'bank' / name) == digest, name
for name, digest in bank['knowledge_inputs'].items():
    assert sha(V2 / name) == digest, name
for name in ('ae_dict.json', 'oe_dict.json'):
    assert sha(V2 / name) == sha(ROOT / 'clip_knowledge_probe_20261010_r2/assets' / name)
    assert sha(V2 / name) == sha(TRAIN / 'source/assets' / name)
bf_manifest = js(sources['bf_manifest.json']['path'])
assert sha(sources['bf_knowledge']['path']) == bf_manifest['knowledge_sha256']
assert sha(sources['bf_bank.pt']['path']) == bf_manifest['bank_sha256']
checks.extend([{'check': name, 'passed': True} for name in (
    'UT-v2 source manifest', 'frozen GloVe builder matches cache SHA256',
    'cache manifest aliases identical', 'EKR bank output hashes',
    'EKR bank knowledge_inputs match local UT-v2 sources',
    'CLIP and TRAIN source AE/OE byte-identical to UT-v2', 'B/F source and bank hashes')])

tables = {}
rows = []
for d in ('ut-zap50k', 'mit-states', 'cgqa'):
    m = js(V1 / d / 'manifest.json')
    with (V1 / d / 'entry_provenance.csv').open(encoding='utf-8-sig', newline='') as f:
        kinds = Counter(r['quality'] for r in csv.DictReader(f))
    c = m['counts']
    rows.append([d, c['attributes'], c['objects'], c['parents'], c['objects'] * 3,
                 c['attributes'] * 3, '<br>'.join(k + '：' + str(v) for k, v in sorted(kinds.items()))])
tables['source_counts'] = table(['数据集', '属性', '对象', 'OI 父类', 'OE 句数', 'AE 句数', '逐条来源记录'], rows)
with Path(sources['v1_summary']['path']).open(encoding='utf-8-sig', newline='') as f:
    rows = [[r['dataset'] + '-v1', f"{float(r['template_K_mean']):.6f}",
             f"{float(r['template_K_population_std']):.6f}",
             f"{100 * float(r['description_OOV_fraction']):.6f}%", r['description_fallback_sentences']] for r in csv.DictReader(f)]
r = js(COND / 'cache/report.json')
rows.append(['UT-v2', f"{r['K_oa_text']['mean']:.6f}", f"{r['K_oa_text']['population_std']:.6f}", f"{100 * r['description_OOV_fraction']:.6f}%", r['description_fallback_sentences']])
tables['encoding_stats'] = table(['资产', '模板 K 均值', '模板 K 总体标准差', '描述 token OOV', '回退句数'], rows)
labels = js(COND / 'bank/labels.json')
ae, oe = js(V2 / 'ae_dict.json'), js(V2 / 'oe_dict.json')
for axis, descriptions in (('attrs', ae), ('objs', oe)):
    tables['v2_' + axis] = table(['ID', '原始标签', '描述 1', '描述 2', '描述 3'],
                               [[i, x] + descriptions[x] for i, x in enumerate(labels[axis])])
bf = js(sources['bf_knowledge']['path'])
tables['bf_groups'] = table(['bank 父类 ID', '父类短语', '子类（数据集原标签）'],
                          [[i, p, '<br>'.join(bf['oi'][p])] for i, p in enumerate(bf_manifest['parents'])])
for axis, texts in (('attrs', bf['ae']), ('objs', bf['oe'])):
    tables['bf_' + axis] = table(['ID', '原始标签', '描述 1', '描述 2', '描述 3'],
                               [[i, x] + texts[x] for i, x in enumerate(labels[axis])])
mask = js(sources['sg_mask']['path'])
tables['mask'] = table(['属性 ID', '标签', '保留 slot（0 为名称）', '被排除的 AE slot'],
                       [[r['attribute_id'], r['attribute'], ', '.join(map(str, r['selected_slots'])),
                         ', '.join(str(i) for i in (1, 2, 3) if i not in r['selected_slots']) or '无'] for r in mask['rows']])
tables['lineage'] = table(['文件', 'UT-v1 SHA256', 'UT-v2 SHA256', '语义变化'],
                         [[name, '`' + sha(V1 / 'ut-zap50k' / name) + '`', '`' + sha(V2 / name) + '`',
                           '内容未改；序列化字节改变' if name == 'oi_dict.json' else 'TRAIN 审查后的描述改写']
                          for name in ('oi_dict.json', 'oe_dict.json', 'ae_dict.json')])
assert js(V1 / 'ut-zap50k/oi_dict.json') == js(V2 / 'oi_dict.json')
tables['asset_hashes'] = table(['关键资产', 'SHA256（本地实际文件）'],
                             [[link(k), '`' + sources[k]['sha256'] + '`'] for k in (
                                 'glove_builder', 'generate.py', 'build_text_prior.py', 'v2_manifest.json',
                                 'bank_bank.pt', 'bank_manifest.json', 'bf_knowledge', 'bf_bank.pt', 'sg_mask', 'clip_protocol', 'train_manifest')])
tables['source_index'] = table(['本地证据文件（点击打开）', '来源 / 用途'],
                             [[link(k), v['role']] for k, v in sources.items()])

template = Path(__file__).with_name('knowledge_history_template.md').read_text(encoding='utf-8')
markdown = re.sub(r'\{\{L:([^}]+)\}\}', lambda m: link(m[1]), template)
markdown = re.sub(r'\{\{T:([^}]+)\}\}', lambda m: tables[m[1]], markdown)
assert not re.search(r'\{\{[LT]:', markdown)
assert markdown.count('$$') % 2 == 0
links = re.findall(r'\]\(</(C:[^>]+)>\)', markdown)
assert all(Path(p).is_file() for p in links)
for source in sources.values():
    assert sha(source['path']) == source['sha256'], source['path']
OUTPUT.write_text(markdown, encoding='utf-8', newline='\n')
receipt = {'document': str(OUTPUT), 'document_sha256': sha(OUTPUT), 'source_files': sources,
           'checks': checks, 'local_links_checked': len(links), 'sources_unchanged': True,
           'existing_full_report_unchanged': True, 'new_training_or_knowledge_generation': False,
           'scope': 'Retrospective source/manifest/script audit; no full tensor-array or GPU metric replay.'}
OUTPUT.with_suffix('.sources.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'output': str(OUTPUT), 'sources': len(sources), 'links': len(links),
                  'checks': len(checks), 'lines': len(markdown.splitlines()), 'bytes': OUTPUT.stat().st_size}, ensure_ascii=False))
