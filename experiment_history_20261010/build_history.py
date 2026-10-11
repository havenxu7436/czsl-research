"""Generate a single retrospective Markdown from frozen local reports/data.

No training, array loading, external requests, or execution of downloaded code.
Authored narrative lives in history_template.md; tables come from stdlib reads.
"""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'output' / '截至20261010_CZSL全部实验_公式修正_决策与数据分析总档案.md'
RECEIPT = OUT.with_suffix('.sources.json')
UT = Path('C:/Users/llenovo/Desktop/czsl_experiment/UT_description_review/UT_v2_results_20261007_153654/files/experiments/semantic_v2/reports/paired_v2')
SOURCES = {
    'original': Path('C:/Users/llenovo/Desktop/0601origin/ZSCL实验设计0601.md'),
    'old_history': ROOT / 'output/截至20261007_CPF_KCC_语义归因与EKR完整实验记录.md',
    'early': ROOT / 'output/week5_evidence_20261005_verified/tables/all_val_metrics.csv',
    'early_epochs': ROOT / 'output/week5_evidence_20261005_verified/tables/epoch_curves.csv',
    'early_test': ROOT / 'output/week5_evidence_20261005_verified/tables/historical_baseline_test_reference.csv',
    'early_bias': ROOT / 'output/week5_evidence_20261005_verified/bias0_ow/bias0_summary.csv',
    'early_audit': ROOT / 'output/week5_evidence_20261005_verified/报告核对说明.md',
    'lambda': ROOT / 'output/week5_evidence_20261005_verified/tables/lambda_analysis.csv',
    'v1_summary': ROOT / '第六步_归因门槛未通过_复核与EKR决策.md',
    'train_review': ROOT / 'ut_semantic_v2_20261006/UT审核完成报告.md',
    'knowledge_v1_stats': ROOT / 'output/week5_evidence_20261005_verified/tables/knowledge_summary.csv',
    'knowledge_v2_cache': ROOT / 'output/BF_conditional_preflight_20261009/extracted/conditional/cache/report.json',
    'knowledge_v2_bank': ROOT / 'output/BF_conditional_preflight_20261009/extracted/conditional/bank/report.json',
    'v2': UT / 'runs.csv',
    'v2_delta': UT / 'paired_summary.csv',
    'v2_review': ROOT / 'output/ut_v2_results_audit_20261007/UT_v2结果审阅与下一步_20261007.md',
    'v2_predictions': ROOT / 'output/ut_v2_prediction_analysis_20261007/UT_v2逐图预测分析与EKR判断_20261007.md',
    'name': ROOT / 'output/name_ekr_pilot_review_20261007/final_metrics.csv',
    'name_epochs': ROOT / 'output/name_ekr_pilot_review_20261007/epoch_metrics.csv',
    'name_review': ROOT / 'output/name_ekr_pilot_review_20261007/名称锚点EKR_阶段B审阅与下一步判断_20261007.md',
    'name_diag': ROOT / 'output/name_ekr_mechanism_review_20261007/名称锚点EKR_机制诊断审阅与下一步_20261007.md',
    'name_diag_data': ROOT / 'output/name_ekr_mechanism_review_20261007/analysis_summary.csv',
    'name_router': ROOT / 'name_anchor_ekr_20261007/transition/name_router.py',
    'centered_router': ROOT / 'ut_semantic_v2_20261006/transition/router.py',
    'align_formula': ROOT / 'ekr_v2_anchor_alignment_20261007/README.md',
    'align3': ROOT / 'output/ekr_v2_review_20261008/artifacts/reports/results.csv',
    'align3_epochs': ROOT / 'output/ekr_v2_review_20261008/epoch_analysis.csv',
    'support': ROOT / 'output/ekr_train_support_review_20261008/independent_audit.json',
    'support_formula': ROOT / 'ekr_train_support_probe_20261008/probe.py',
    'sg': ROOT / 'output/sgekr_review_20261008/artifacts/reports/results.csv',
    'sg_epochs': ROOT / 'output/sgekr_review_20261008/epoch_analysis.csv',
    'sg_mask': ROOT / 'output/sgekr_review_20261008/artifacts/support_mask.json',
    'sg_mask_code': ROOT / 'sgekr_support_mask_20261008/pilot.py',
    'sg_review': Path('C:/Users/llenovo/Desktop/czsl_experiment/20261007_key_归档/SG-EKR审阅与下一步实践方案_20261008.md'),
    'prototype': ROOT / 'output/sgekr_review_20261008/train_pair_holdout_conditional_mix_probe.csv',
    'prototype_add': ROOT / 'output/sgekr_review_20261008/train_pair_holdout_additive_probe.csv',
    'prototype_folds': ROOT / 'output/sgekr_review_20261008/train_image_fold_prototype_reference.csv',
    'epoch_revision': ROOT / 'output/ekr_v2_budget10_20261008/Epoch回溯审查与v2十轮续训方案_20261008.md',
    'align10': ROOT / 'output/ekr_v2_budget10_review_20261008/artifacts/reports/results.csv',
    'align10_epochs': ROOT / 'output/ekr_v2_budget10_review_20261008/artifacts/reports/epoch_curves.csv',
    'align10_review': ROOT / 'output/ekr_v2_budget10_review_20261008/EKR_v2十轮续训审阅与下一步_20261008.md',
    'bf': ROOT / 'output/BF_server_results_20261008/BF_seed3351_实际训练与结果.csv',
    'bf_review': ROOT / 'output/BF_server_results_20261008/BF四组结果分析_20261009.md',
    'bf_manual': ROOT / 'output/LogiCZSL论文方法重建_CPF-BF训练完整手册_20261008.md',
    'bf_logic': ROOT / 'output/BF_conditional_preflight_20261009/extracted/bf/source/models/bf_logic_loss.py',
    'bf_adapter': ROOT / 'logiczsl_cpf_bf_20261008/model/bf_logic.py',
    'bf_preflight': ROOT / 'output/BF审计与条件属性StageA实践命令_20261009.md',
    'stagea': ROOT / 'output/conditional_attribute_stageA_results_20261009/StageA审阅与五组条件属性训练命令_20261009.md',
    'head_formula': ROOT / 'output/conditional_attribute_stageA_results_20261009/条件属性Head_实现公式与代码说明_20261009.md',
    'checkpoint_fix': ROOT / 'output/conditional_attribute_stageA_results_20261009/Torch1.13兼容错误说明与r2重跑命令_20261009.md',
    'pilot': ROOT / 'output/conditional_attribute_pilot_results_20261009_r2/extracted/pilot/summary.json',
    'pilot_review': ROOT / 'output/conditional_attribute_pilot_results_20261009_r2/Pilot完整审阅与下一步判断_20261009.md',
    'mix_replay': ROOT / 'output/conditional_attribute_pilot_results_20261009_r2/下一步修正方案_温和概率更新与同结构知识对照_20261009.md',
    'mix_data': ROOT / 'output/conditional_attribute_pilot_results_20261009_r2/fixed_mixture_full_epoch_replay.json',
    'generic': ROOT / 'output/generic_mixture_controls_results_20261010/extracted/controls/summary.json',
    'generic_review': ROOT / 'output/generic_mixture_controls_results_20261010/温和概率混合Generic对照_完整审阅与下一步判断_20261010.md',
    'goals': ROOT / 'output/成功标准分层与实验路线回溯修正_20261010.md',
    'repro': ROOT / 'output/0601原方案复现完成度与剩余任务_20261010.md',
    'route': ROOT / 'output/原方案路径3_视觉知识监督路线修订_20261010.md',
    'clip_fix': ROOT / 'output/CLIP探针划分检查修复_r2_操作方法_20261010.md',
    'clip_prompts': ROOT / 'clip_knowledge_probe_20261010_r2/prompts.py',
    'clip': ROOT / 'output/clip_knowledge_probe_results_20261010_r2/independent_audit.json',
    'clip_review': ROOT / 'output/clip_knowledge_probe_results_20261010_r2/CLIP知识证据探针_r2_独立复核与下一步判断_20261010.md',
    'sentence': ROOT / 'output/train_knowledge_audit_results_20261010/extracted/result/sentence_train_report.json',
    'pooled': ROOT / 'output/train_knowledge_audit_results_20261010/extracted/result/pooled_train_report.json',
    'train_audit': ROOT / 'output/train_knowledge_audit_results_20261010/independent_audit.json',
    'train_audit_review': ROOT / 'output/train_knowledge_audit_results_20261010/TRAIN知识落地审查_独立复核与路线判断_20261010.md',
}
USED = {}


def read(key):
    path = SOURCES[key]
    data = path.read_bytes()
    USED[path.resolve().as_posix()] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    return data.decode('utf-8-sig')


def rows(key):
    return list(csv.DictReader(read(key).splitlines()))


def js(key):
    return json.loads(read(key))


def f(value):
    if value is None or value == '':
        return '—'
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return '{:.6f}'.format(value)
    return str(value).replace('|', '\\|').replace('\n', ' ')


def table(headers, data):
    # Right-align numeric columns in the Markdown itself, not only the HTML preview.
    numeric = re.compile(r'^[+−-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+−-]?\d+)?(?:%)?$')
    alignment = []
    for index in range(len(headers)):
        cells = [f(row[index]) for row in data if f(row[index]) != '—']
        alignment.append('---:' if cells and all(numeric.fullmatch(cell) for cell in cells) else '---')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(alignment) + ' |'] +
                     ['| ' + ' | '.join(f(x) for x in row) + ' |' for row in data])


def csv_table(key, columns, headers=None):
    data = rows(key)
    assert data and all(c in data[0] for c in columns), (key,columns)
    def cast(x):
        if isinstance(x, str) and re.fullmatch(r'-?\d+', x):
            return int(x)
        try:
            return float(x)
        except (TypeError, ValueError):
            return x
    return table(headers or columns, [[cast(r.get(c, '')) for c in columns] for r in data])


def selected_table(data, baseline):
    base_cw, base_ow = 100 * baseline['cw']['AUC'], 100 * baseline['ow']['AUC']
    out = [['冻结 C_text', 0, base_cw, 100 * baseline['cw']['best_hm'], base_ow,
            100 * baseline['ow']['best_hm'], 0.0, 0.0]]
    for name, result in data.items():
        s = result['selected']
        m = s['metrics_raw']
        cw, ow = 100*m['cw']['AUC'], 100*m['ow']['AUC']
        out.append([name, s['epoch'], cw, 100*m['cw']['best_hm'], ow,
                    100*m['ow']['best_hm'], cw-base_cw, ow-base_ow])
    return table(['组别', '选中epoch', 'CW AUC%', 'CW HM%', 'OW AUC%', 'OW HM%', 'ΔCW pp', 'ΔOW pp'], out)


def histories(directory, variants):
    out = []
    for variant in variants:
        path = directory / 'runs' / variant / 'history.json'
        data = path.read_bytes()
        USED[path.resolve().as_posix()] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        history = json.loads(data)
        assert history['completed_epoch'] == 10
        assert len(history['ledger']) == 11
        for item in history['ledger']:
            m = item['metrics_raw']
            out.append([variant, item['epoch'], 100*m['cw']['AUC'], 100*m['cw']['best_hm'],
                        100*m['ow']['AUC'], 100*m['ow']['best_hm']])
    return table(['组别', 'epoch', 'CW AUC%', 'CW HM%', 'OW AUC%', 'OW HM%'], out)


def build_tables():
    t = {}
    v1 = next(row for row in rows('knowledge_v1_stats') if row['dataset'] == 'ut-zappos')
    v2 = js('knowledge_v2_cache')
    bank = js('knowledge_v2_bank')
    t['KNOWLEDGE_ASSETS'] = table(
        ['版本', 'K 均值', 'K 总体标准差', '描述词元 OOV 率（%）', '整句回退次数'],
        [['UT v1', float(v1['template_K_mean']), float(v1['template_K_population_std']),
          100 * float(v1['description_OOV_fraction']), int(v1['description_fallback_sentences'])],
         ['UT v2', v2['K_oa_text']['mean'], v2['K_oa_text']['population_std'],
          100 * v2['description_OOV_fraction'], v2['description_fallback_sentences']]])
    oov = []
    for axis, label in [('a', '属性描述'), ('o', '对象描述')]:
        entry = bank[axis]
        oov.append([label, entry['oov_tokens'], entry['total_tokens'],
                    100 * entry['oov_tokens'] / entry['total_tokens']])
    total_oov = sum(bank[axis]['oov_tokens'] for axis in ['a', 'o'])
    total_tokens = sum(bank[axis]['total_tokens'] for axis in ['a', 'o'])
    assert abs(total_oov / total_tokens - v2['description_OOV_fraction']) < 1e-15
    oov.append(['合计', total_oov, total_tokens, 100 * total_oov / total_tokens])
    t['KNOWLEDGE_OOV'] = table(['UT v2 描述类别', 'OOV 词元数', '词元总数', 'OOV 率（%）'], oov)
    t['KNOWLEDGE_ERRORS'] = table(
        ['数值检查项', '属性 bank', '对象 bank'],
        [['聚合与 cache 的最大绝对误差', bank['a']['old_C_max_abs_error'], bank['o']['old_C_max_abs_error']],
         ['中心化残差和的最大绝对值（近似）',
          format(bank['a']['residual_sum_max_abs'], '.10f'),
          format(bank['o']['residual_sum_max_abs'], '.10f')]])
    t['EARLY'] = csv_table('early', ['dataset','seed','variant','setting','auc_pct','hm_pct','best_seen_pct','best_unseen_pct','best_epoch'],
                          ['数据集','seed','组','设置','AUC%','HM%','best-seen%','best-unseen%','最佳epoch'])
    t['TEST'] = csv_table('early_test', ['dataset','seed','setting','auc_pct','hm_pct','best_seen_pct','best_unseen_pct'],
                         ['数据集','seed','设置','AUC%','HM%','best-seen%','best-unseen%'])
    t['EARLY_BIAS'] = csv_table('early_bias', ['dataset','group','n','A_pair_accuracy_pct','C_pair_accuracy_pct','net_corrected_images'],
                               ['数据集','分组','图像数','A pair%','C pair%','净纠正数'])
    t['LAMBDA'] = csv_table('lambda', ['diagnostic','lambda','cw_auc_pct','cw_hm_pct','ow_auc_pct','ow_hm_pct','eligible','selected'],
                           ['后处理','λ','CW AUC%','CW HM%','OW AUC%','OW HM%','保护项合格','被选择'])
    t['V2'] = csv_table('v2', ['seed','variant','setting','best_epoch','auc_pct','hm_pct','best_seen_pct','best_unseen_pct'],
                       ['seed','组别','设置','选中epoch','AUC%','HM%','best-seen%','best-unseen%'])
    t['V2_DELTAS'] = csv_table('v2_delta', ['comparison','setting','delta_auc_pp_mean','delta_auc_pp_sample_std','delta_hm_pp_mean','delta_unseen_pp_mean'],
                              ['配对差','设置','ΔAUC均值pp','ΔAUC样本std','ΔHM均值pp','ΔUmax均值pp'])
    vr = {(int(r['seed']),r['variant'],r['setting']):r for r in rows('v2')}
    vd = []
    for seed in [0,3351,2026]:
        def delta(control, setting):
            return float(vr[seed,'C_text',setting]['auc_pct'])-float(vr[seed,control,setting]['auc_pct'])
        vd.append([seed,delta('A_unit','cw'),delta('C_shuffle','cw'),delta('A_unit','ow'),delta('C_shuffle','ow')])
    t['V2_SEED_DELTAS'] = table(['seed','正确−单位名称CW pp','正确−错配CW pp','正确−单位名称OW pp','正确−错配OW pp'],vd)
    for key, label in [('name','NAME'),('align3','ALIGN3'),('align10','ALIGN10'),('sg','SG')]:
        ep = 'best_epoch' if key == 'name' else 'selected_epoch'
        t[label] = csv_table(key, ['variant','setting',ep,'auc_pct','hm_pct'],
                             ['组别','设置','选中epoch','AUC%','HM%'])
    t['BF'] = csv_table('bf', ['variant','setting','AUC','best_hm','best_seen','best_unseen','best_epoch'],
                       ['组别','设置','AUC%','HM%','best-seen%','best-unseen%','最佳epoch'])
    t['PROTO_FOLDS'] = csv_table('prototype_folds', ['held_fold','method','n_images','known_object_attribute_accuracy_pct'],
                                ['留出半折','方法','图像数','GT-object属性准确率%'])
    t['PROTO_ALL'] = csv_table('prototype', ['beta','case','pair_accuracy_pct','mean_ce','corrected_vs_anchor','broken_vs_anchor'],
                              ['混合β','证据','组合准确率%','CE','纠正','破坏'])
    t['PROTO_ADD'] = csv_table('prototype_add', ['gain','pair_accuracy_pct','mean_ce','corrected_vs_anchor','broken_vs_anchor'],
                              ['加性gain','组合准确率%','CE','纠正','破坏'])
    p = js('pilot')
    g = js('generic')
    assert p['formal_optimizer_steps'] == 4500
    assert g['new_formal_optimizer_steps'] == 2700
    t['PILOT'] = selected_table(p['results'], p['baseline_metrics_raw'])
    t['GENERIC'] = selected_table(g['results'], p['baseline_metrics_raw'])
    mix = js('mix_data')
    assert mix['protocol']['new_optimizer_steps'] == 0 and mix['protocol']['alpha'] == .25
    t['MIX_SELECTED'] = selected_table({k:{'selected':v} for k,v in mix['selected'].items()},p['baseline_metrics_raw'])
    mix_rows = []
    for variant,ledger in mix['records'].items():
        assert len(ledger) == 11
        for item in ledger:
            m = item['metrics_raw']
            mix_rows.append([variant,item['epoch'],100*m['cw']['AUC'],100*m['cw']['best_hm'],100*m['ow']['AUC'],100*m['ow']['best_hm']])
    assert len(mix_rows) == 55
    t['MIX_EPOCHS'] = table(['原head','epoch','25%读出CW AUC%','CW HM%','OW AUC%','OW HM%'],mix_rows)
    t['NAME_DIAGNOSTICS'] = csv_table('name_diag_data', ['view','case_id','native_auc_pct','native_hm_pct','pair_pct','net_correct','margin_delta_mean','gold_null_pct'],
                                     ['视图','干预','AUC%','HM%','bias0 pair%','相对锚点净纠正','margin均值差','GT null%'])
    c = js('clip')
    assert c['passed'] and c['all_top1_micro_macro_confusion_replays_match']
    ct = []
    for name, item in c['metrics'].items():
        tr, va = item['train']['all'], item['val']
        ct.append([name,tr['accuracy_pct'],tr['pair_macro_accuracy_pct'],va['all']['accuracy_pct'],
                   va['seen']['accuracy_pct'],va['unseen']['accuracy_pct'],va['unseen']['pair_macro_accuracy_pct']])
    t['CLIP'] = table(['配方','TRAIN micro%','TRAIN pair macro%','VAL all micro%','VAL seen micro%','VAL unseen micro%','VAL unseen pair macro%'],ct)
    s = js('sentence')
    st = []
    for family, views in s['families'].items():
        for view in views:
            correct = 'T_K' if family == 'OE_AE' else 'T_AE'
            for name, m in view['variants'].items():
                st.append([family,view['ae_index']+1,
                           view['oe_index']+1 if view['oe_index'] is not None else '不含OE',
                           name,m['micro_pct'],m['pair_macro_pct'],m['correct']])
    assert len(st) == 36
    t['SENTENCE_ALL'] = table(['配方','AE位置(1起)','OE位置(1起)','内容','TRAIN micro%','TRAIN pair macro%','判对张数'],st)
    pooled = js('pooled')
    name = pooled['per_attribute']
    ae3 = s['families']['AE_only'][2]['variants']['T_AE']['per_attribute']
    at = []
    for n,m in zip(name,ae3):
        assert n['attr'] == m['attr'] and n['images'] == m['images']
        nc = n['variants']['T_name']['correct']
        mc = round(m['images']*m['micro_pct']/100)
        at.append([n['attr'],n['images'],nc,mc,mc-nc])
    assert sum(x[-1] for x in at) == 3772
    t['AE3_ATTRS'] = table(['属性','TRAIN图数','名称判对','AE第三条判对','净变化'],at)
    overview = []
    for key,label in [('T_name','名称'),('T_K','正确OE＋AE均值'),('T_AE','正确AE三条均值')]:
        m = pooled['metrics'][key]
        overview.append([label,m['micro_pct'],m['pair_macro_pct'],m['correct']])
    for i,view in enumerate(s['families']['AE_only']):
        m = view['variants']['T_AE']
        overview.append(['AE第{}条'.format(i+1),m['micro_pct'],m['pair_macro_pct'],m['correct']])
    for key,label in [('T_AE_perm1','AE第3条perm1'),('T_AE_perm2','AE第3条perm2')]:
        m = s['families']['AE_only'][2]['variants'][key]
        overview.append([label,m['micro_pct'],m['pair_macro_pct'],m['correct']])
    t['TRAIN_AUDIT_OVERVIEW'] = table(['TRAIN配方','micro%','pair macro%','判对张数'],overview)
    # Preserve the rounded logged trajectories, not invented full-precision values.
    bf_epochs, bf_losses = [], []
    for variant in ['A_ref','B','E_ref','F']:
        path = ROOT / 'output/BF_conditional_preflight_20261009/extracted/bf/runs' / variant / 'train.log'
        raw = path.read_bytes()
        USED[path.resolve().as_posix()] = {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        log = raw.decode('utf-8-sig')
        matches = re.findall(r'Val/Test Epoch (\d+): AUC=([\d.]+) \| best_hm=([\d.]+) \| best_seen=([\d.]+) \| best_unseen=([\d.]+)',log)
        assert len(matches) == 10 and [int(m[0]) for m in matches] == list(range(1,11)), variant
        for epoch,auc,hm,seen,unseen in matches:
            bf_epochs.append([variant,int(epoch),100*float(auc),100*float(hm),100*float(seen),100*float(unseen)])
        last_by_epoch = {}
        for match in re.finditer(r'BF_LOGIC epoch=(\d+) iter=(\d+) cpf=([\d.eE+\-]+) oi=([\d.eE+\-]+) oe=([\d.eE+\-]+) ae=([\d.eE+\-]+) weighted=([\d.eE+\-]+)',log):
            epoch,iteration,*values = match.groups()
            last_by_epoch[int(epoch)] = [variant,int(epoch),int(iteration),*[float(x) for x in values]]
        if variant in ['B','F']:
            assert len(last_by_epoch) == 10
            bf_losses.extend(last_by_epoch.values())
    t['BF_EPOCHS'] = table(['组','epoch','日志CW AUC%','日志CW HM%','日志best-seen%','日志best-unseen%'],bf_epochs)
    t['BF_LOSSES'] = table(['组','epoch','末次logic打印iter','该batch CPF loss','OI','OE','AE','加权logic'],bf_losses)
    for key,label in [('early_epochs','EARLY_EPOCHS'),('name_epochs','NAME_EPOCHS'),('align10_epochs','ALIGN_EPOCHS'),('sg_epochs','SG_EPOCHS')]:
        fields = ['variant','epoch','loss_mean','cw_auc_pct','cw_hm_pct']
        headers = ['组','epoch','训练loss','CW AUC%','CW HM%']
        if key == 'early_epochs':
            fields = ['dataset','variant','epoch','loss_last_logged_avg','auc_pct','hm_pct']
            headers = ['数据集','组','epoch','末次日志loss均值','日志AUC%','日志HM%']
        t[label] = csv_table(key,fields,headers)
    pilot_dir = ROOT / 'output/conditional_attribute_pilot_results_20261009_r2/extracted/pilot'
    generic_dir = ROOT / 'output/generic_mixture_controls_results_20261010/extracted/controls'
    t['PILOT_EPOCHS'] = histories(pilot_dir, ['P_name','P_K','P_generic','P_perm1','P_perm2'])
    # Correct generic trajectory was reused, its mixed ledger is recorded separately.
    t['GENERIC_EPOCHS'] = histories(generic_dir, ['G_name','G_perm1','G_perm2'])
    gk = generic_dir / 'runs/G_K/reference_replays'
    candidates = sorted(gk.glob('**/*.json'))
    replay_rows = []
    for path in candidates:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if 'metrics_raw' in data and 'epoch' in data:
            USED[path.resolve().as_posix()] = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
            m = data['metrics_raw']
            replay_rows.append(['G_K',data['epoch'],100*m['cw']['AUC'],100*m['cw']['best_hm'],100*m['ow']['AUC'],100*m['ow']['best_hm']])
    assert len(replay_rows) == 11, 'Expected all eleven reused mixed G_K evaluations'
    t['GK_EPOCHS'] = table(['组别','epoch','CW AUC%','CW HM%','OW AUC%','OW HM%'],sorted(replay_rows,key=lambda r:r[1]))
    return t


def main():
    if OUT.exists() or RECEIPT.exists():
        if not OUT.is_file() or not RECEIPT.is_file():
            raise SystemExit('Preserve existing unmatched deliverable; choose a new version.')
        prior = json.loads(RECEIPT.read_text(encoding='utf-8'))
        if prior.get('sha256') != hashlib.sha256(OUT.read_bytes()).hexdigest() or prior.get('scope') != 'local retrospective compilation only; no new experimental execution':
            raise SystemExit('Preserve externally modified deliverable; choose a new version.')
    template = Path(__file__).with_name('history_template.md').read_text(encoding='utf-8')
    tables = build_tables()
    def source(match):
        key = match.group(1)
        read(key)
        path = SOURCES[key].resolve()
        return '[{}](<{}>)'.format(path.name, '/' + path.as_posix())
    final = re.sub(r'\{\{SOURCE:([a-z0-9_]+)\}\}',source,template)
    for name,value in tables.items():
        final = final.replace('{{TABLE:'+name+'}}',value)
    assert '{{' not in final, 'Unresolved document placeholder'
    for target in re.findall(r'\]\(<(/C:/[^>]+)>\)',final):
        assert Path(target[1:]).is_file(), target
    assert final.count('$$') % 2 == 0
    manifest_table = table(['本次实际读取的数据/资料','SHA256'],
                           [['[{}](<{}>)'.format(Path(path).name,'/'+path),item['sha256']]
                            for path,item in sorted(USED.items())])
    final += '\n\n### 附录 H：本总档的数据来源指纹\n\n' + manifest_table + '\n'
    OUT.write_text(final,encoding='utf-8',newline='\n')
    receipt = {'scope':'local retrospective compilation only; no new experimental execution',
               'generated_md':OUT.resolve().as_posix(),'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),
               'source_files':USED,'table_count':len(tables),'local_links_checked':True,
               'formula_double_dollar_balanced':True,'training_started':False}
    receipt['authored_inputs'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__),Path(__file__).with_name('history_template.md')]}
    RECEIPT.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'file':OUT.as_posix(),'bytes':OUT.stat().st_size,'lines':len(final.splitlines()),
                      'source_files':len(USED),'tables':len(tables)},ensure_ascii=False))


if __name__ == '__main__':
    main()
